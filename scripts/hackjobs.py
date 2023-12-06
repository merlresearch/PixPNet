# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os.path as osp
import re
from pprint import pprint

import pandas as pd
import torch
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as F
from torchmetrics import Accuracy

proj_path = osp.dirname(osp.dirname(osp.realpath(__file__)))

try:
    import pixpnet
except ImportError:
    import sys

    sys.path.append(proj_path)

    import pixpnet  # noqa: F401
finally:
    from pixpnet.data import get_metadata
    from pixpnet.lightning.lightning_data import LitData
    from pixpnet.protonets.models import feature_extractor_zoo as zoo
    from pixpnet.utils import now_str


class LitModel(LightningModule):
    def __init__(self, hackjob_model: zoo.ModelHackjob, classifier_only=False):
        super().__init__()
        self.model = hackjob_model
        self.classifier_only = classifier_only
        if self.classifier_only:
            for p in hackjob_model.parameters():
                p.requires_grad = False
            self.model.module_dict.eval()
            if hackjob_model.requires_training:
                for p in hackjob_model.classifier.parameters():
                    p.requires_grad = True

        acc_kws = {"task": "multiclass", "num_classes": hackjob_model.num_classes}
        self.accuracy = Accuracy(**acc_kws)
        self.accuracy_top5 = Accuracy(top_k=5, **acc_kws)
        if self.model.multi_output:
            self.test_accuracy = torch.nn.ModuleDict(
                {name: Accuracy(**acc_kws) for name in self.model.last_module_name}
            )
            self.test_accuracy_top5 = torch.nn.ModuleDict(
                {name: Accuracy(top_k=5, **acc_kws) for name in self.model.last_module_name}
            )
        else:
            self.test_accuracy = Accuracy(**acc_kws)
            self.test_accuracy_top5 = Accuracy(top_k=5, **acc_kws)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx=None):
        x, y = batch
        pred = self(x)
        if self.model.multi_output:
            pred = [*pred.values()]
            loss = sum(F.cross_entropy(pred_layer, y) for pred_layer in pred)
            self.accuracy(pred[-1], y)
        else:
            loss = F.cross_entropy(pred, y)
            self.accuracy(pred, y)
        self.log("train_loss", loss)
        self.log("train_accuracy__last", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx=None, dataset_idx=None):
        x, y = batch
        pred = self(x)
        metric_kwargs = dict(on_step=True, on_epoch=True, sync_dist=True)
        if self.model.multi_output:
            for name in self.model.last_module_name:
                self.test_accuracy[name](pred[name], y)
                self.test_accuracy_top5[name](pred[name], y)
                self.log(f"{name}__test_accuracy", self.test_accuracy[name], **metric_kwargs)
                self.log(f"{name}__test_accuracy_top5", self.test_accuracy_top5[name], **metric_kwargs)
        else:
            self.test_accuracy(pred, y)
            self.test_accuracy_top5(pred, y)
            self.log(f"{self.model.last_module_name}__test_accuracy", self.test_accuracy, **metric_kwargs)
            self.log(f"{self.model.last_module_name}__test_accuracy_top5", self.test_accuracy_top5, **metric_kwargs)

    def configure_optimizers(self):
        assert self.model.classifier is not None
        params = (
            self.model.classifier_parameters()
            if self.classifier_only
            else [{"params": p["params"], "lr": 1e-3} for p in self.model.classifier_parameters()]
            + [{"params": self.model.feature_extractor_parameters(), "lr": 1e-3}]
        )
        return torch.optim.Adam(params, lr=0.001)


def pretty_key(key: str):
    return re.sub(r"(^[a-z]|[_ ][a-z])", lambda m: m.group().upper().replace("_", " "), key)


# noinspection PyPep8Naming
class config:
    debug = False

    # noinspection PyPep8Naming
    class dataset:
        needs_unaugmented = False
        val_size = 0  # no tuning, so no validation set
        name = "imagenette"
        augment_factor = 1
        root = None

    # noinspection PyPep8Naming
    class train:
        batch_size = 32

    # noinspection PyPep8Naming
    class test:
        batch_size = 32


def evaluate_hackjob(model_name, data, epochs=2):
    print("Get metadata")
    metadata = get_metadata(config)
    num_classes = metadata.output_size

    joint = False
    print(f"joint = {joint}")
    predict_module_names = None
    idx = 0
    tot = 1
    all_results = []

    while idx < tot:
        print("Load feature extractor")
        base_model = zoo.get_feature_extractor_base(model_name, pretrained=True)

        if predict_module_names is None:
            predict_module_names = base_model.predict_module_names
            if joint:
                predict_module_names = [predict_module_names]
            tot = len(predict_module_names)
        predict_module_name = predict_module_names[idx]

        print(f"predict_module_name={predict_module_name}")

        print(f"Create hackjob model for {model_name}")
        hackjob_model = zoo.ModelHackjob(
            base_model,
            last_module_name=predict_module_name,
            num_classes=num_classes,
        )

        print("Create trainer")
        trainer = Trainer(
            accelerator="auto",
            max_epochs=epochs,
            devices=1 if torch.cuda.is_available() else None,
        )

        print("Create LitModel")
        model = LitModel(hackjob_model)

        if hackjob_model.requires_training:
            print("we need to train")
            trainer.fit(model, data)

        print("evaluation time")
        predict_module_name_ = [predict_module_name] if isinstance(predict_module_name, str) else predict_module_name
        result_all_layers = {}
        for k, v in trainer.test(model, data)[0].items():
            layer, metric = k.split("__")
            assert layer in predict_module_name_
            result_all_layers[layer] = result_all_layers.get(layer, {})
            result_all_layers[layer][pretty_key(metric)] = v

        mod_seq_list = [*base_model.module_sequence.keys()]
        for layer_name in predict_module_name_:
            result = result_all_layers[layer_name]
            result["Dataset"] = config.dataset.name
            result["Height"] = metadata.input_size
            result["Width"] = metadata.input_size
            result["Stage #"] = mod_seq_list.index(layer_name) + 1
            result["Stage Name"] = layer_name
            result["Model"] = model_name
            all_results.append(result)

        idx += 1

    pprint(all_results)

    return all_results


def run():
    import argparse
    import os
    import shutil

    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", default=None)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    result_dir = os.path.join(os.path.dirname(__file__), "../results/hackjob")
    if args.debug:
        config.dataset.name = "CIFAR10"
        result_dir += "_DEBUGGING"

    os.makedirs(result_dir, exist_ok=True)

    print("Get datasets")
    data = LitData(config)

    all_results = []
    for model_name in zoo.supported_models:
        if args.filter and args.filter not in model_name:
            continue
        all_results.extend(evaluate_hackjob(model_name, data, epochs=args.epochs))

    save_path = os.path.join(result_dir, f"hackjob_results_{now_str()}.csv")
    print(save_path)
    pd.DataFrame(all_results).to_csv(save_path, index=False)

    if args.debug:
        shutil.rmtree(result_dir)


if __name__ == "__main__":
    run()
