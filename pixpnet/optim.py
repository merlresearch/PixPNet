# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import inspect
import re
from typing import Any, Dict, Optional, Set, Tuple, Type

import torch
from pytorch_warmup import ExponentialWarmup
from pytorch_warmup.base import BaseWarmup
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, StepLR

from pixpnet.utils import get_logger, intersect_func_and_kwargs

logger = get_logger(__name__)

_OPTIMIZER_MAP = {attr: getattr(torch.optim, attr) for attr in dir(torch.optim) if attr != "Optimizer"}
_OPTIMIZER_MAP = {attr: cls for attr, cls in _OPTIMIZER_MAP.items() if inspect.isclass(cls)}
_LOOSE_OPTIMIZER_MAP = {}
for _attr, _cls in _OPTIMIZER_MAP.items():
    _attr_split = re.split(r"(?=(?<!^)[A-Z][a-z]|(?<![A-Z])[A-Z]$)", _attr)
    _attr_lower = "".join(map(str.lower, _attr_split))
    _attr_lower_ = "_".join(map(str.lower, _attr_split))
    if _attr_lower in _LOOSE_OPTIMIZER_MAP or _attr_lower_ in _LOOSE_OPTIMIZER_MAP:
        _cls_existing = _LOOSE_OPTIMIZER_MAP.get(_attr_lower, _LOOSE_OPTIMIZER_MAP.get(_attr_lower_))
        raise RuntimeError(
            f"{_attr_lower} already in optimizers! Overlapping class names in "
            f"lowercase was unexpected and cannot be resolved: "
            f"{_cls_existing} and {_cls}"
        )
    _LOOSE_OPTIMIZER_MAP[_attr_lower] = _cls
    if _attr_lower != _attr_lower_:
        _LOOSE_OPTIMIZER_MAP[_attr_lower_] = _cls


def get_optimizer_cls(
    config: argparse.Namespace,
    ignore: Optional[Set[str]] = None,
) -> Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]:
    if ignore is None:
        ignore = set()
    try:
        optimizer_cls = _LOOSE_OPTIMIZER_MAP[config.optimizer.name.lower()]
    except KeyError:
        raise ValueError(f'No such optimizer "{config.optimizer.name}"')
    hparams, invalid_keys = intersect_func_and_kwargs(
        optimizer_cls,
        config.optimizer,
        exclude_func_args={"params"},
        exclude_kwargs={"name", "throttle_lr", "lr_schedule", "lr_scheduler", "lr_factor", "warmup_period"} | ignore,
    )
    if invalid_keys:
        logger.warning(
            f"Will not pass the following invalid optimizer "
            f"hyperparameters to {optimizer_cls.__name__}: "
            f'{", ".join(invalid_keys)}'
        )
    logger.info(f"Optimizer hyperparameters for {optimizer_cls.__name__}: " f"{hparams}")
    return optimizer_cls, hparams


class LRWithWarmupMixin:
    def __init__(self, *args, **kwargs):
        self.warmup_scheduler: BaseWarmup = kwargs.pop("warmup")
        super().__init__(*args, **kwargs)

    def step(self, epoch=None):
        if self.warmup_scheduler is None:
            super().step(epoch=epoch)
        else:
            with self.warmup_scheduler.dampening():
                super().step(epoch=epoch)


class MultiStepLRWithWarmup(LRWithWarmupMixin, MultiStepLR):
    """"""


class CosineAnnealingLRWithWarmup(LRWithWarmupMixin, CosineAnnealingLR):
    """"""


class StepLRWithWarmup(LRWithWarmupMixin, StepLR):
    """"""


def get_scheduler(optimizer: torch.optim.Optimizer, config: argparse.Namespace) -> LRWithWarmupMixin:
    """"""
    if config.optimizer.warmup_period:
        lr_warmup = ExponentialWarmup(optimizer, warmup_period=config.optimizer.warmup_period)
    else:
        lr_warmup = None
    if config.optimizer.lr_scheduler == "multistep":
        lr_scheduler = MultiStepLRWithWarmup(
            optimizer,
            milestones=config.optimizer.lr_schedule,
            gamma=config.optimizer.lr_factor,
            last_epoch=-1,
            warmup=lr_warmup,
        )
    elif config.optimizer.lr_scheduler == "step":
        assert len(config.optimizer.lr_schedule) == 1, config.optimizer.lr_schedule
        lr_scheduler = StepLRWithWarmup(
            optimizer,
            step_size=config.optimizer.lr_schedule[0],
            gamma=config.optimizer.lr_factor,
            last_epoch=-1,
            warmup=lr_warmup,
        )
    elif config.optimizer.lr_scheduler == "cosine":
        lr_scheduler = CosineAnnealingLRWithWarmup(
            optimizer,
            T_max=config.train.epochs,
            eta_min=0,
            last_epoch=-1,
            warmup=lr_warmup,
        )
    else:
        raise NotImplementedError(f"Scheduler {config.optimizer.lr_scheduler}")

    return lr_scheduler
