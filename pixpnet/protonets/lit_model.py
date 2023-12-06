# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
from typing import Tuple

import torch
from torch import nn
from torchmetrics import Accuracy

try:
    from pytorch_lightning.loops import FitLoop
except ImportError:
    from pytorch_lightning.loops.fit_loop import _FitLoop as FitLoop

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from pixpnet.data import get_metadata
from pixpnet.lightning.lightning_data import LitData
from pixpnet.lightning.lit_module import BaseLitModel
from pixpnet.optim import get_optimizer_cls, get_scheduler
from pixpnet.protonets.loss import ClusterLoss, L1ReadoutLoss, SeparationLoss
from pixpnet.protonets.models.protonet import ProtoNet, protonet
from pixpnet.protonets.push import push_prototypes
from pixpnet.utils import get_logger, intersect_func_and_kwargs

logger = get_logger(__name__)


def params_with_grad(parameters):
    return filter(lambda p: p.requires_grad, parameters)


def make_optimizers_proto(
    model: ProtoNet,
    config: argparse.Namespace,
) -> Tuple[torch.optim.Optimizer, ...]:
    """"""
    optimizer_cls, hparams = get_optimizer_cls(config, ignore={"fine_tune_lr", "readout_lr"})

    readout_params = None
    if model.last_layer is not None:
        readout_params = [
            {
                "params": params_with_grad(model.last_layer.parameters()),
                "lr": config.optimizer.readout_lr,
                "weight_decay": 0,
            },
        ]
    all_params = [
        # feature extractor
        {"params": params_with_grad(model.features.parameters()), "lr": config.optimizer.fine_tune_lr},
        # add on layers
        {"params": params_with_grad(model.add_on_layers.parameters())},
        # prototype layers
        {"params": params_with_grad([model.prototype_vectors]), "weight_decay": 0},
    ]
    readout_optimizer = None
    if readout_params is not None:
        all_params += readout_params
        readout_optimizer = optimizer_cls(params=readout_params, **hparams)

    optimizer = optimizer_cls(params=all_params, **hparams)
    return optimizer, readout_optimizer


def _set_grad(model, features=True, add_on_layers=True, prototype_vectors=True, last_layer=True):
    for p in model.features.parameters():
        p.requires_grad = features
    for p in model.add_on_layers.parameters():
        p.requires_grad = add_on_layers
    model.prototype_vectors.requires_grad = prototype_vectors
    if model.last_layer is not None:
        for p in model.last_layer.parameters():
            p.requires_grad = last_layer


def last_only(model):
    _set_grad(model, features=False, add_on_layers=False, prototype_vectors=False)


def warm_only(model):
    _set_grad(model, features=False)


def joint(model):
    _set_grad(model)


class ProtoLitModel(BaseLitModel):
    def __init__(self, config, feature_extractor=None):
        super().__init__(config)
        metadata = get_metadata(config)
        self.num_classes = metadata.output_size
        self.input_size = metadata.input_size
        hparams, invalid_keys = intersect_func_and_kwargs(
            protonet,
            config.model,
            exclude_func_args={"num_classes"},
            exclude_kwargs={"name"},
        )
        if invalid_keys:
            logger.warning(
                f"Will not pass the following invalid model "
                f"hyperparameters to {protonet.__name__}: "
                f'{", ".join(invalid_keys)}'
            )
        logger.info(f"Model hyperparameters for {protonet.__name__}: " f"{hparams}")
        if feature_extractor is not None:
            logger.info(
                f"feature_extractor is not None, ignoring config " f'option of {hparams.get("feature_extractor")}'
            )
            hparams["feature_extractor"] = feature_extractor
        self.model = protonet(num_classes=self.num_classes, input_size=self.input_size, **hparams)

        self.lr_scheduler = None
        self.readout_optimizer = self.lr_scheduler_configs = None

        # losses
        self.xent = self._metric_per_split(nn.CrossEntropyLoss)
        class_specific = self.config.model.class_specific
        self.l1 = self._metric_per_split(L1ReadoutLoss, class_specific=class_specific)
        self.cluster = self._metric_per_split(ClusterLoss, class_specific=class_specific)
        self.separation = self._metric_per_split(SeparationLoss) if class_specific else None

        # metrics
        acc_kws = {"task": "multiclass", "num_classes": self.num_classes}
        self.accuracy = self._metric_per_split(Accuracy, **acc_kws)
        self.weighted_accuracy = self._metric_per_split(Accuracy, average="macro", **acc_kws)

    def _forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def configure_optimizers(self):
        if self.optimizer is None:
            self.optimizer, self.readout_optimizer = make_optimizers_proto(self.model, self.config)
            self.lr_scheduler = get_scheduler(self.optimizer, self.config)
        else:
            logger.warning(
                "In configure_optimizers: will not reinitialize "
                "optimizer(s) and schedulers as self.optimizer is "
                "not None"
            )
        return [self.optimizer], [
            {
                "scheduler": self.lr_scheduler,
                "interval": "epoch",
            }
        ]

    def _use_optimizer(self, optimizer, lr_scheduler_configs):
        self.trainer.optimizers = [optimizer]
        self.trainer.strategy.lr_scheduler_configs = lr_scheduler_configs
        self.trainer.optimizer_frequencies = []

    def use_readout_optimizer(self):
        if self.lr_scheduler_configs is None:
            self.lr_scheduler_configs = self.trainer.lr_scheduler_configs
        self._use_optimizer(self.readout_optimizer, [])

    def use_default_optimizer(self):
        self._use_optimizer(self.optimizer, self.lr_scheduler_configs)

    def _shared_eval(self, batch, batch_idx, dataset_idx, prefix):
        coefs = self.config.loss

        if len(batch) == 2:
            x, y = batch
        else:
            # first element is an index
            _, x, y = batch

        result = self(x)
        if self.training:
            self.log_result(result)

        logits = result["logits"]
        min_distances = result["min_distances"]

        # maybe free mem of unused tensors
        del result

        # compute losses
        loss = 0
        if coefs.xent != 0:
            xent_loss = self.xent[prefix](logits, y)
            self.log(f"{prefix}_xent", xent_loss)
            xent_loss *= coefs.xent
            self.log(f"{prefix}_xent_weighted", xent_loss)
            loss += xent_loss

        if coefs.cluster != 0:
            cluster_args = (min_distances, y, self.model)
            cluster_loss = self.cluster[prefix](*cluster_args)
            self.log(f"{prefix}_cluster_loss", cluster_loss)
            cluster_loss *= coefs.cluster
            self.log(f"{prefix}_cluster_loss_weighted", cluster_loss)
            loss += cluster_loss

        if self.separation is not None and coefs.separation != 0:
            *sep_args, sep_kwargs = (min_distances, y, self.model, {"return_avg": True})
            separation_loss = self.separation[prefix](*sep_args, **sep_kwargs)
            separation_loss, avg_separation_loss = separation_loss
            self.log(f"{prefix}_avg_separation_loss", avg_separation_loss)

            self.log(f"{prefix}_separation_loss", separation_loss)
            separation_loss *= coefs.separation
            self.log(f"{prefix}_separation_loss_weighted", separation_loss)
            loss += separation_loss

        if self.model.last_layer is not None and coefs.l1 != 0 and self.model.readout_type in {"linear", "sparse"}:
            l1_loss = self.l1[prefix](self.model)
            self.log(f"{prefix}_l1_loss", l1_loss)
            l1_loss = l1_loss * coefs.l1  # no in-place modification
            self.log(f"{prefix}_l1_loss_weighted", l1_loss)
            loss += l1_loss

        self.log(f"{prefix}_total_loss", loss)

        self.accuracy[prefix](logits, y)
        self.log(f"{prefix}_accuracy", self.accuracy[prefix], prog_bar=True)
        self.weighted_accuracy[prefix](logits, y)
        self.log(f"{prefix}_weighted_accuracy", self.weighted_accuracy[prefix])

        return loss

    @property
    def tb_logger(self):
        for pl_logger in self.loggers:
            if isinstance(pl_logger, TensorBoardLogger):
                return pl_logger

    def log_result(self, result):
        if not self.config.tb_logging:
            return
        pl_logger = self.tb_logger
        if pl_logger is None:
            logger.warning("Could not find TB logger...")
            return
        global_step = self.global_step
        for k, v in result.items():
            self.safe_add_histogram(pl_logger, k, v, global_step)

    @staticmethod
    def safe_add_histogram(pl_logger: TensorBoardLogger, name, *args, **kwargs):
        try:
            pl_logger.experiment.add_histogram(name, *args, **kwargs)
        except ValueError as e:
            logger.warning(f"Error when logging name={name}: {e}")

    def on_after_backward(self):
        if not self.config.tb_logging:
            return
        pl_logger = self.tb_logger
        if pl_logger is None:
            logger.warning("Could not find TB logger...")
            return
        global_step = self.global_step
        for name, module in self.model.add_on_layers.named_modules(prefix="add_on_layers"):
            if not isinstance(module, nn.Conv2d):
                continue
            self.safe_add_histogram(pl_logger, name, module.weight, global_step)
            if module.weight.grad is not None:
                self.safe_add_histogram(pl_logger, name + "_grad", module.weight.grad, global_step)
        self.safe_add_histogram(pl_logger, "prototype_vectors", self.model.prototype_vectors, global_step)
        if self.model.prototype_vectors.grad is not None:
            self.safe_add_histogram(pl_logger, "prototype_vectors_grad", self.model.prototype_vectors.grad, global_step)


class ProtoFitLoop(FitLoop):
    # Potential alt/other resource:
    # https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/finetuning-scheduler.html
    def __init__(self, config, data_module=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        assert (data_module is None) == (not self.config.train.push_prototypes)
        self.data_module = data_module
        self._repeated_advance = False
        self._n_pushes = 0

    @property
    def model(self) -> ProtoNet:
        return self.pl_module.model

    @property
    def pl_module(self) -> ProtoLitModel:
        return self.trainer.lightning_module

    def on_advance_start(self) -> None:
        if self._repeated_advance:
            return
        logger.info("on_advance_start")
        super().on_advance_start()

        current_epoch = self.trainer.current_epoch + 1
        if current_epoch <= self.config.optimizer.warmup_period:
            logger.info(
                f"warm_only: "
                f"trainer.current_epoch={self.trainer.current_epoch} "
                f"config.optimizer.warmup_period="
                f"{self.config.optimizer.warmup_period}"
            )
            warm_only(self.model)
        else:
            logger.info(
                f"joint: "
                f"trainer.current_epoch={self.trainer.current_epoch} "
                f"config.optimizer.warmup_period="
                f"{self.config.optimizer.warmup_period}"
            )
            joint(self.model)

    def on_advance_end(self) -> None:
        current_epoch = self.trainer.current_epoch + 1
        is_first_epoch = current_epoch == 1
        is_last_epoch = current_epoch == self.max_epochs
        if self.config.train.push_prototypes and (
            current_epoch % self.config.train.push_every == 0 or is_last_epoch or is_first_epoch
        ):
            self._n_pushes += 1
            logger.info(
                f"Pushing prototypes (Push {self._n_pushes} " f"on epoch {current_epoch} / {self.config.train.epochs})"
            )

            run_push(
                trainer=self.trainer,
                model=self.pl_module,
                data=self.data_module,
                config=self.config,
            )
        super().on_advance_end()


def run_push(
    trainer: Trainer,
    model: LightningModule,
    data: LitData,
    config,
):
    push_prototypes(
        data.train_no_aug_dataloader(),
        model.model,
        class_specific=config.model.class_specific,
        duplicate_filter=config.train.push_duplicate_filter,
    )

    readout_push_epochs = config.train.readout_push_epochs
    if readout_push_epochs and model.model.readout_type != "proto" and model.model.last_layer is not None:
        # now that we have pushed prototypes, we need to fine-tune the
        #  last layer
        last_only(model.model)
        # disable validation during last-layer training
        limit_val_batches = trainer.limit_val_batches
        trainer.limit_val_batches = 0
        # switch to readout optimizer (no lr schedule)
        model.use_readout_optimizer()
        # convex optimization of readout
        for _ in range(readout_push_epochs):
            trainer.fit_loop.on_advance_start()
            trainer.fit_loop.advance()
        # switch back to default optimizer
        model.use_default_optimizer()
        # re-enable validation
        trainer.limit_val_batches = limit_val_batches
