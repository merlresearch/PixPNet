# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os.path as osp
import sys
import unittest

proj_path = osp.dirname(osp.dirname(osp.realpath(__file__)))

try:
    import scripts
except ImportError:
    sys.path.append(proj_path)

    import scripts  # noqa: F401
finally:
    from scripts.hackjobs import run as run_hackjobs
    from scripts.run_protonet import main as run_protonet


def test_run_protonet():
    argv_orig = sys.argv.copy()
    sys.argv.extend(["-c", "configs/protonets/proto-cifar10-dummy.yaml"])
    run_protonet()
    sys.argv = argv_orig


def test_hackjobs():
    argv_orig = sys.argv.copy()
    sys.argv.extend(["--epochs", "1", "--filter", "vgg11_bn", "--debug"])
    run_hackjobs()
    sys.argv = argv_orig


if __name__ == "__main__":
    unittest.main()
