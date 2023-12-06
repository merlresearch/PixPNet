# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from argparse import Namespace
from glob import glob
from os import path as osp
from typing import Tuple

from pixpnet.protonets.lit_model import ProtoLitModel
from pixpnet.utils import parse_config_file


def load_config_and_best_model(logdir) -> Tuple[Namespace, ProtoLitModel]:
    results_dir = osp.join(logdir, "results")
    config = parse_config_file(osp.join(results_dir, "config.yaml"))

    best_model_path_txt = osp.join(results_dir, "best_model_path.txt")
    best_model_path_txt_exists = osp.isfile(best_model_path_txt)
    best_model_path_exists = True
    if best_model_path_txt_exists:
        with open(best_model_path_txt, "r") as f:
            best_model_path = f.read().strip()
            if not osp.isabs(best_model_path):
                best_model_path = osp.join(results_dir, best_model_path)
        if not osp.isfile(best_model_path):
            best_model_path_exists = False
            print(f"WARN: {best_model_path} does not exist!")
    if not best_model_path_txt_exists or not best_model_path_exists:
        model_paths = glob(osp.join(logdir, "**", "*.ckpt"), recursive=True)
        if len(model_paths) != 1:
            raise ValueError(f"{len(model_paths)} paths found but expected 1")
        best_model_path = model_paths[0]

    model: ProtoLitModel = ProtoLitModel.load_from_checkpoint(best_model_path, config=config)

    return config, model
