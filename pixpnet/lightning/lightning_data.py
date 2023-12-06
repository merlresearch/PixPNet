# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from pixpnet.data import get_datasets
from pixpnet.utils import get_logger, num_cpus

logger = get_logger(__name__)


class LitData(LightningDataModule):
    def __init__(self, config, num_workers=None, **kwargs):
        super().__init__()
        self.config = config
        self.train = self.train_no_aug = self.val = self.test = None
        self.kwargs = kwargs
        # Required to check if setup was called prior...
        # https://github.com/Lightning-AI/lightning/issues/9865
        self.datasets_loaded = False
        if num_workers is None:
            num_workers = num_cpus()
        self.num_workers = num_workers

    def setup(self, stage=None):
        """called on every GPU"""
        if self.datasets_loaded:
            return

        logger.info(f"Loading the {self.config.dataset.name} dataset " f"(val_size={self.config.dataset.val_size})")

        datasets = get_datasets(self.config, **self.kwargs)

        if self.config.dataset.needs_unaugmented:
            self.train, self.train_no_aug, self.val, self.test = datasets
        else:
            self.train, self.val, self.test = datasets

        # get_datasets may modify val_size
        if self.config.dataset.val_size == 0:
            if self.trainer:
                self.trainer.limit_val_batches = 0
                self.trainer.num_sanity_val_steps = 0

        self.datasets_loaded = True

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def train_no_aug_dataloader(self):
        if not self.config.dataset.needs_unaugmented:
            raise ValueError("Unaugmented train data set requested, but " "--dataset.needs-unaugmented is False")
        return DataLoader(
            self.train_no_aug,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.config.test.batch_size, num_workers=self.num_workers, drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.config.test.batch_size, num_workers=self.num_workers, drop_last=False
        )
