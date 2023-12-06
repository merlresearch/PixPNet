# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import time
from abc import ABCMeta, abstractmethod
from typing import Any

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

from pixpnet.utils import get_logger

logger = get_logger(__name__)


class CollisionlessModuleDict(nn.ModuleDict):
    __slots__ = ()

    __internal_prefix = "_no_collision__"

    def _correct_key(self, key):
        return self.__internal_prefix + key

    def _uncorrect_key(self, key):
        return key[len(self.__internal_prefix) :]

    def __getitem__(self, key: str) -> nn.Module:
        key = self._correct_key(key)
        return super().__getitem__(key)

    def __setitem__(self, key: str, module: nn.Module) -> None:
        key = self._correct_key(key)
        super().__setitem__(key, module)

    def __delitem__(self, key: str) -> None:
        key = self._correct_key(key)
        super().__delitem__(key)

    def __contains__(self, key: str) -> bool:
        key = self._correct_key(key)
        return super().__contains__(key)

    def pop(self, key: str) -> nn.Module:
        r"""Remove key from the ModuleDict and return its module.

        Args:
            key (string): key to pop from the ModuleDict
        """
        key = self._correct_key(key)
        return super().pop(key)

    def keys(self):
        r"""Return an iterable of the ModuleDict keys."""
        return tuple(map(self._uncorrect_key, super().keys()))

    def items(self):
        r"""Return an iterable of the ModuleDict key/value pairs."""
        return tuple(zip(self.keys(), self.values()))


class BaseLitModel(LightningModule, metaclass=ABCMeta):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.optimizer = None

        # training stats
        self._train_time_total = 0
        self._train_time_per_epoch = 0
        self._actual_epoch_count = 0
        self._infer_count = 0
        self._infer_batch_count = 0
        self._inference_time_per_sample = 0
        self._inference_time_per_batch = 0
        self._train_t0 = None
        self._inference_t0 = None

    @property
    def train_time_total(self):
        return self._train_time_total

    @property
    def train_time_per_epoch(self):
        return self._train_time_per_epoch

    @property
    def inference_time_per_sample(self):
        return self._inference_time_per_sample

    @property
    def inference_time_per_batch(self):
        return self._inference_time_per_batch

    @abstractmethod
    def _forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @staticmethod
    def _metric_per_split(metric, *args, **kwargs):
        return CollisionlessModuleDict(
            {"train": metric(*args, **kwargs), "val": metric(*args, **kwargs), "test": metric(*args, **kwargs)}
        )

    def forward(self, x, *args, **kwargs) -> Any:
        if not self.training:
            # only record inference time in non-training mode
            self._inference_t0 = time.time()
        out = self._forward(x, *args, **kwargs)
        if not self.training:
            duration = time.time() - self._inference_t0
            self._inference_time_per_batch = (self._inference_time_per_batch * self._infer_batch_count + duration) / (
                self._infer_batch_count + 1
            )
            self._infer_batch_count += 1
            self._inference_time_per_sample = (self._inference_time_per_sample * self._infer_count + duration) / (
                self._infer_count + len(x)
            )
            self._infer_count += len(x)
        return out

    def on_train_start(self):
        if self.config.debug:
            torch.autograd.set_detect_anomaly(True)
        hp_lr_metrics = {f"hp/lr_group_{i}": 0 for i in range(len(self.optimizer.param_groups))}
        for lit_logger in self.loggers:
            args = (hp_lr_metrics,) if isinstance(lit_logger, TensorBoardLogger) else ()
            lit_logger.log_hyperparams(self.config.optimizer, *args)
            lit_logger.log_hyperparams(self.config.train)
            lit_logger.log_hyperparams(self.config.model)

    def on_train_epoch_start(self) -> None:
        self._train_t0 = time.time()

    def on_train_epoch_end(self) -> None:
        duration = time.time() - self._train_t0
        self._train_time_total += duration
        # running mean
        self._train_time_per_epoch = (self._train_time_per_epoch * self._actual_epoch_count + duration) / (
            self._actual_epoch_count + 1
        )
        self._actual_epoch_count += 1

    def training_step(self, batch, batch_idx, dataset_idx=None):
        loss = self._shared_eval(batch, batch_idx, dataset_idx, "train")
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.log(f"hp/lr_group_{i}", param_group["lr"])
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=None):
        self._shared_eval(batch, batch_idx, dataset_idx, "val")

    def test_step(self, batch, batch_idx, dataset_idx=None):
        self._shared_eval(batch, batch_idx, dataset_idx, "test")

    @abstractmethod
    def _shared_eval(self, batch: Any, batch_idx: int, dataset_idx: int, prefix: str) -> torch.Tensor:
        """
        Handle batch, compute forward, compute loss and other metrics,
        then return the loss.
        """
        raise NotImplementedError
