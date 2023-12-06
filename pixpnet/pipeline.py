# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from time import sleep

import git
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.utilities.model_summary import summarize

from pixpnet.lightning.lightning_data import LitData
from pixpnet.macs import get_macs
from pixpnet.utils import get_logger, nested_ns_to_nested_dict, now_str, pretty_si_units, yaml_dump

logger = get_logger(__name__)


def create_trainer(config):
    """ """
    # Stop training if loss becomes NaN
    # Infinite patience as we don't want to actually stop early due to loss
    # (e.g., fluctuations due to pushing prototypes)
    callbacks = [EarlyStopping("train_total_loss", check_finite=True, patience=float("inf"))]
    if config.train.checkpoint:
        logger.warning(
            "If you are replacing prototypes, loading this "
            "checkpoint will not load the model with replaced "
            "prototypes!"
        )
        checkpoint_callback = ModelCheckpoint(
            monitor=None,
        )
        callbacks.append(checkpoint_callback)
    else:
        checkpoint_callback = None
    if config.train.stochastic_weight_averaging:
        callbacks.append(StochasticWeightAveraging())

    logger.info(f'Logging checkpoints to "{config.log_dir}"')
    kwargs = {}
    if config.train.hparam_tune:
        # Currently, this feature supports two modes: ‘power’ scaling and
        # ‘binsearch’ scaling. In ‘power’ scaling, starting from a batch size
        # of 1 keeps doubling the batch size until an out-of-memory (OOM) error
        # is encountered. Setting the argument to ‘binsearch’ will initially
        # also try doubling the batch size until it encounters an OOM, after
        # which it will do a binary search that will finetune the batch size.
        # Additionally, it should be noted that the batch size scaler cannot
        # search for batch sizes larger than the size of the training dataset.
        kwargs["auto_scale_batch_size"] = "binsearch"

    loggers = [
        CSVLogger(save_dir=config.log_dir, name="csv"),
        TensorBoardLogger(save_dir=config.log_dir, default_hp_metric=False, name="tensorboard"),
    ]

    if config.gpus is None:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            if os.environ["CUDA_VISIBLE_DEVICES"].strip():
                config.gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
            else:
                config.gpus = 0
            gpu_set_using = "$CUDA_VISIBLE_DEVICES"
        else:
            config.gpus = int(bool(torch.cuda.device_count()))
            gpu_set_using = "`torch.cuda.device_count()`"
        logger.info(f"Set --gpus to {config.gpus} using {gpu_set_using}")

    if config.train.val_every_n_epoch is None:
        # validate ~50 times as a default
        config.train.val_every_n_epoch = max(1, round(config.train.epochs / 50))

    kwargs = {"num_sanity_val_steps": 2 if config.debug else 0}
    if config.dataset.val_size == 0:
        kwargs = {"limit_val_batches": 0, "num_sanity_val_steps": 0}

    accumulate_grad_batches = getattr(config.train, "accumulate_grad_batches", None)
    if accumulate_grad_batches:
        kwargs["accumulate_grad_batches"] = accumulate_grad_batches
    trainer = Trainer(
        default_root_dir=config.log_dir,
        callbacks=callbacks,
        devices=config.gpus if config.gpus > 0 else 1,  # 1 CPU proc. if no GPU
        accelerator="gpu" if config.gpus > 0 else "cpu",
        # DistributedDataParallel if using more than 1 GPU
        strategy="auto" if config.gpus < 2 else "ddp",
        max_epochs=config.train.epochs,
        logger=loggers,
        enable_model_summary=False,
        check_val_every_n_epoch=config.train.val_every_n_epoch,
        benchmark=True,  # note caveats about determinism
        gradient_clip_val=config.train.gradient_clip_norm,
        gradient_clip_algorithm="norm",
        profiler=config.profiler if config.profile else None,
        log_every_n_steps=50,
        **kwargs,
    )
    return trainer, checkpoint_callback


def load_checkpoint_patient(
    LitModel,
    model_path,
    config,
    retries=100,
    sleep_secs=3,
    **kwargs,
):
    while retries:
        retries -= 1
        try:
            model = LitModel.load_from_checkpoint(model_path, config=config, **kwargs)
        except FileNotFoundError:
            logger.warning(
                f'Could not find "{model_path}"! Sleeping for '
                f"{sleep_secs} seconds - hopefully filesystem "
                f"synchronizes by then...This warning is far more "
                f"common when DDP strategy is used on older "
                f"filesystems."
            )
            sleep(sleep_secs)
        else:
            return model
    raise FileNotFoundError(
        f'No model found at path "{model_path}" after ' f"{retries * sleep_secs} seconds! Has it been " f"deleted?"
    )


def run(config, LitModel, FitLoop=None, train_loop=None, lit_model_kws=None):
    suffix = ("__" + config.run_id) if config.run_id else ""
    config.log_dir = os.path.join(
        config.log_dir, config.dataset.name.lower() + suffix, config.model.name.lower(), now_str()
    )
    if config.debug:
        logger.info("We are in debug mode.")
        config.log_dir += "_DEBUG"

    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
    except git.InvalidGitRepositoryError:
        sha = None
    config.git_commit_hash = sha

    log_dir_orig = config.log_dir
    retry = 0
    while True:
        try:
            os.makedirs(config.log_dir)
        except FileExistsError:
            retry += 1
            config.log_dir = f"{log_dir_orig}_{retry}"
        else:
            break
    if retry:
        logger.warning(f'"{log_dir_orig}" already existed, so we elected to ' f'log to "{config.log_dir}" instead.')

    seed_everything(config.seed, workers=True)

    if config.test.batch_size is None:
        config.test.batch_size = config.train.batch_size
        logger.info(f"Set config.test.batch_size = {config.test.batch_size}")

    logger.info("Create model.")
    if lit_model_kws is None:
        lit_model_kws = {}
    model = LitModel(config, **lit_model_kws)

    logger.info("Create data module.")
    data = LitData(config)

    logger.info("Create trainer.")
    trainer, checkpoint_callback = create_trainer(config)

    if FitLoop is not None:
        try:
            from pytorch_lightning.loops import FitLoop as _  # noqa: F401

            fl_kwargs = {}
        except ImportError:
            fl_kwargs = {"trainer": trainer}

        if config.dataset.needs_unaugmented:
            fl_kwargs["data_module"] = data
        fit_loop = FitLoop(
            config=config, min_epochs=trainer.fit_loop.min_epochs, max_epochs=trainer.fit_loop.max_epochs, **fl_kwargs
        )
        trainer.fit_loop = fit_loop

    if config.train.hparam_tune:
        logger.info("Tuning hyperparameters before training.")
        trainer.tune(model)

    logger.info("Train model.")
    if train_loop is None:
        trainer.fit(model, data)
    else:
        train_loop(trainer=trainer, model=model, data=data, config=config)

    # model summary (flops and memory)
    summary = summarize(model, max_depth=2)
    logger.info(summary)
    # macs
    x, y = next(iter(data.train_dataloader()))
    macs = get_macs(model, x.to(device=model.device))
    logger.info(f"MAC Operations = {pretty_si_units(macs)}")

    model_info = dict(
        model_size=summary.model_size,
        trainable_parameters=summary.trainable_parameters,
        total_parameters=summary.total_parameters,
        macs=macs,
        train_time_total=model.train_time_total,
        train_time_per_epoch=model.train_time_per_epoch,
        inference_time_per_sample=model.inference_time_per_sample,
        inference_time_per_batch=model.inference_time_per_batch,
    )
    logger.info(f"All model info:\n{model_info}")

    best_model_path = None
    if checkpoint_callback is not None:
        best_model_path = checkpoint_callback.best_model_path
        logger.info(f'Load weights of best model from "{best_model_path}"')
        model = load_checkpoint_patient(LitModel, best_model_path, config=config, **lit_model_kws)

    logger.info("Test the best model.")
    results = trainer.test(model, data)

    results_dir = os.path.join(config.log_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "scores.yaml")
    config_path = os.path.join(results_dir, "config.yaml")
    model_info_path = os.path.join(results_dir, "model_info.yaml")
    best_model_path_txt = os.path.join(results_dir, "best_model_path.txt")

    logger.info(f"Writing config to {config_path}")
    with open(config_path, "w") as f:
        yaml_dump(nested_ns_to_nested_dict(config), f, indent=4)

    logger.info(f"Writing results to {results_path}")
    with open(results_path, "w") as f:
        yaml_dump(results, f, indent=4)

    logger.info(f"Writing model info to {model_info_path}")
    with open(model_info_path, "w") as f:
        yaml_dump(model_info, f, indent=4)

    if best_model_path is not None:
        with open(best_model_path_txt, "w") as f:
            f.write(os.path.relpath(best_model_path, results_dir))

    return model
