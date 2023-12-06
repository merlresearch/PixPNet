# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import inspect
import re
import sys
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Union

import torchvision
from packaging import version
from torch import nn
from torch.nn.modules import activation
from torch.nn.modules.batchnorm import _NormBase
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd  # noqa: F401
from torch.nn.modules.padding import _ConstantPadNd, _ReflectionPadNd, _ReplicationPadNd
from torch.nn.modules.pooling import (
    _AdaptiveAvgPoolNd,
    _AdaptiveMaxPoolNd,
    _AvgPoolNd,
    _LPPoolNd,
    _MaxPoolNd,
    _MaxUnpoolNd,
)
from torchvision import models
from torchvision.models import _api

from pixpnet.utils import get_logger, load_module_copy

major, minor = sys.version_info[:2]
if major > 3 or (major == 3 and minor >= 9):
    OrderedDict_T = OrderedDict
else:
    OrderedDict_T = Dict

tv_has_registration_mech = hasattr(_api, "BUILTIN_MODELS")
if tv_has_registration_mech:
    BUILTIN_MODELS_ORIG = _api.BUILTIN_MODELS
    _api.BUILTIN_MODELS = {}

resnet = load_module_copy("torchvision.models.resnet")
densenet = load_module_copy("torchvision.models.densenet")
vgg = load_module_copy("torchvision.models.vgg")
inception = load_module_copy("torchvision.models.inception")
squeezenet = load_module_copy("torchvision.models.squeezenet")

if tv_has_registration_mech:
    _api.BUILTIN_MODELS = BUILTIN_MODELS_ORIG

logger = get_logger(__name__)

activation_layers = tuple(
    k
    for k in vars(activation).values()
    if (inspect.isclass(k) and issubclass(k, nn.Module) and k.__module__ == activation.__name__)
)
pooling_layers = (_MaxUnpoolNd, _AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd, _MaxPoolNd, _AvgPoolNd, _LPPoolNd)
padding_layers = (_ReplicationPadNd, _ConstantPadNd, _ReflectionPadNd)
channel_invariant_layers = (
    pooling_layers
    + padding_layers
    + (
        nn.CrossMapLRN2d,
        nn.LayerNorm,
    )
)


def sequential_to_dict(seq: nn.Sequential, prefix=None, last=None) -> OrderedDict:
    prefix = (prefix + "_") if prefix else ""
    d = defaultdict(lambda: 0)
    seq_dict = OrderedDict()
    for i, layer in enumerate(seq):
        name = type(layer).__name__.lower()
        d[name] += 1
        if last and i + 1 == len(seq):
            d_name = last
        else:
            d_name = f"{prefix}{name}{d[name]}"
        seq_dict[d_name] = layer
    return seq_dict


class FeatureExtractorMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def module_sequence(self) -> OrderedDict:
        """If I were to iterate over this and call the modules in a sequence, I
        should get the same output as the original model."""
        raise NotImplementedError

    @property
    def predict_module_names(self) -> List[str]:
        raise NotImplementedError

    @property
    def final_feature_extractor(self) -> str:
        raise NotImplementedError

    @property
    def last_layer(self) -> str:
        return next(reversed(self.module_sequence.keys()))


def _ensure_unique(sequence):
    seq_as_set = {*sequence}
    assert len(seq_as_set) == len(sequence), f"Not unique! ({len(seq_as_set)} != {len(sequence)})"
    return sequence


class FeatureExtractorHackjob(nn.Module):
    def __init__(self, model: FeatureExtractorMixin, last_module_name: Union[str, List[str]] = None):
        super().__init__()
        # store model this way to avoid registering all modules (some may not
        #  be used and be incorrectly associated with the model)
        self._model = (model,)
        if last_module_name is None:
            # default to last feature extraction layer
            last_module_name = self.model.final_feature_extractor
        self.multi_output = not isinstance(last_module_name, str)
        self.last_module_name = _ensure_unique(last_module_name) if self.multi_output else last_module_name
        if self.multi_output:
            modules = self._init_multi_output()
        else:
            modules = self._init_single_output()
        self.module_dict = nn.ModuleDict(modules)

    @property
    def model(self) -> FeatureExtractorMixin:
        return self._model[0]

    def _init_single_output(self):
        modules = OrderedDict()
        module_sequence = self.model.module_sequence
        for name, module in module_sequence.items():
            modules[name] = module
            if name == self.last_module_name:
                break
        else:
            raise ValueError(f'Invalid module name "{self.last_module_name}" ' f"(valid: {module_sequence.keys()})")
        return modules

    def _init_multi_output(self):
        last_module_names = {*self.last_module_name}
        visited_last_module_names = set()
        modules = OrderedDict()
        maybe_modules = OrderedDict()
        module_sequence = self.model.module_sequence
        for name, module in module_sequence.items():
            maybe_modules[name] = module
            if name in last_module_names:
                visited_last_module_names.add(name)
                # Only add to modules if they are necessary to get to
                #  proceeding target layers
                modules.update(maybe_modules)
                maybe_modules = OrderedDict()
        if visited_last_module_names != last_module_names:
            raise ValueError(
                f'Invalid module names "'
                f"{last_module_names - visited_last_module_names}"
                f'" (valid: {module_sequence.keys()})'
            )
        return modules

    @property
    def out_channels(self) -> Union[int, OrderedDict_T[str, int]]:
        try:
            names = [*self.module_dict.keys()]
            modules = [*self.module_dict.values()]
            if self.multi_output:
                return OrderedDict(
                    (
                        (module_name, self._compute_out_channels_name(module_name, names, modules))
                        for module_name in self.last_module_name
                    )
                )
            else:
                return self._compute_out_channels_name(self.last_module_name, names, modules)
        except Exception:
            import traceback

            traceback.print_exc()
            raise

    def _compute_out_channels_name(self, output_name: str, names: List[str], modules: List[nn.Module]) -> int:
        if output_name == self.model.last_layer:
            return None  # classifier has no channels
        name_as_idx = names.index(output_name)
        out_channels = FeatureExtractorHackjob._compute_out_channels(modules[: name_as_idx + 1])
        if out_channels is None:
            raise TypeError(f"Could not compute output channels of layer name " f'"{output_name}"!')
        # otherwise
        return out_channels

    @staticmethod
    def _compute_out_channels(modules: Iterable[nn.Module]) -> Union[int, None]:
        modules_reversed = reversed([*modules])
        for module in modules_reversed:
            if isinstance(module, _ConvNd):
                return module.out_channels
            elif isinstance(module, _NormBase):
                return module.num_features
            elif isinstance(module, nn.GroupNorm):
                return module.num_channels
            elif module.__class__.__name__ == "_DenseBlock":
                out_channels_module = 0
                for k, v in module.named_modules():
                    if not bool(re.match(r"^denselayer\d+$", k)):
                        continue
                    if out_channels_module == 0:
                        # for the first denselayer (should have init num
                        #  features which we can infer for other cat input...)
                        out_channels_module += v.conv1.in_channels
                    out_channels_module += v.conv2.out_channels
                return out_channels_module
            elif isinstance(module, (nn.Sequential, nn.ModuleList)):
                out_channels = FeatureExtractorHackjob._compute_out_channels(module)
                if out_channels is not None:
                    return out_channels
                continue
            elif isinstance(module, nn.ModuleDict):
                out_channels = FeatureExtractorHackjob._compute_out_channels([*module.values()])
                if out_channels is not None:
                    return out_channels
                continue
            elif isinstance(module, (activation_layers + channel_invariant_layers)):
                continue
            elif module._modules:
                out_channels = FeatureExtractorHackjob._compute_out_channels([*module._modules.values()])
                if out_channels is not None:
                    return out_channels
                continue
            else:
                raise TypeError(f"Unsupported module type {type(module)}")

    def forward(self, x, last_module_name=None):
        if last_module_name is not None:
            return_single = isinstance(last_module_name, str)
            if return_single:
                last_module_name = [last_module_name]
            out = self._forward_multi(x, last_module_name)
            if return_single:
                return out[last_module_name[0]]
            else:
                return out
        else:
            if self.multi_output:
                return self._forward_multi(x)
            else:
                return self._forward_single(x)

    def _forward_single(self, x):
        for name, module in self.module_dict.items():
            x = module(x)
        return x

    def _forward_multi(self, x, last_module_name=None):
        if last_module_name is None:
            last_module_name = self.last_module_name
        visited = 0
        outputs = OrderedDict()
        for name, module in self.module_dict.items():
            x = module(x)
            if name in last_module_name:
                outputs[name] = x
                visited += 1
                if len(last_module_name) == visited:
                    # break early as we have visited everything requested
                    break
        return outputs


class ModelHackjob(FeatureExtractorHackjob):
    def __init__(self, model: FeatureExtractorMixin, num_classes: int, last_module_name: Union[str, List[str]] = None):
        self.num_classes = num_classes
        super().__init__(model, last_module_name)

    def _init_single_output(self):
        modules = super()._init_single_output()
        module_sequence = self.model.module_sequence
        self.requires_training = True
        if len(modules) < len(module_sequence):
            # Add classifier
            self.classifier = self._make_classifier()
        else:
            last_name, last_module = next(reversed(modules.items()))
            if last_module.out_features != self.num_classes:
                # replace linear
                logger.warning(
                    f"replacing {last_name}={last_module} with new "
                    f"Linear as num_classes differs ("
                    f"{last_module.out_features} != {self.num_classes})"
                )
                self.classifier = self._make_readout(last_module.in_features)
                del modules[last_name]
            else:
                self.classifier = None
                self.requires_training = False
        return modules

    def _init_multi_output(self):
        modules = super()._init_multi_output()
        last_module_names = {*self.last_module_name}
        module_sequence = self.model.module_sequence
        classifier_after = OrderedDict()
        self.requires_training = False
        for i, (name, module) in enumerate(modules.items()):
            if name in last_module_names:
                if i + 1 < len(module_sequence):
                    self.requires_training = True
                    # Add classifier
                    classifier_after[name] = self._make_classifier()
                else:
                    assert isinstance(module, nn.Linear), module
                    if module.out_features != self.num_classes:
                        # replace linear
                        logger.warning(
                            f"replacing {name}={module} with new "
                            f"Linear as num_classes differs "
                            f"({module.out_features} != {self.num_classes})"
                        )
                        self.requires_training = True
                        modules[name] = nn.Identity()
                        classifier_after[name] = self._make_readout(module.in_features)
                    else:
                        classifier_after[name] = nn.Identity()
        self.classifier = nn.ModuleDict(classifier_after)
        return modules

    def _make_classifier(self) -> nn.Sequential:
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            self._make_readout(),
        )

    def _make_readout(self, in_features=None) -> nn.Linear:
        return nn.LazyLinear(self.num_classes) if in_features is None else nn.Linear(in_features, self.num_classes)

    @property
    def out_channels(self) -> int:
        raise TypeError(
            f"out_channels is an invalid property for " f"{type(self)} (the output(s) is from a linear " f"layer(s))"
        )

    def classifier_parameters(self):
        if self.multi_output:
            return [{"params": clf.parameters() for clf in self.classifier.values()}]
        else:
            return [{"params": self.classifier.parameters()}]

    def feature_extractor_parameters(self):
        return self.module_dict.parameters()

    def _forward_single(self, x):
        x = super()._forward_single(x)
        if self.classifier is not None:
            x = self.classifier(x)
        return x

    def _forward_multi(self, x):
        features = super()._forward_multi(x)
        outputs = OrderedDict()
        for name in self.last_module_name:
            outputs[name] = self.classifier[name](features[name])
        return outputs


class ResNet(FeatureExtractorMixin, resnet.ResNet):
    @property
    def module_sequence(self) -> OrderedDict:
        return OrderedDict(
            (
                ("conv1_no_act", self.conv1),
                ("bn1", self.bn1),
                ("conv1", self.relu),
                ("maxpool", self.maxpool),
                ("layer1", self.layer1),
                ("layer2", self.layer2),
                ("layer3", self.layer3),
                ("layer4", self.layer4),
                ("avgpool", self.avgpool),
                ("flatten", nn.Flatten(start_dim=1)),
                ("fc", self.fc),
            )
        )

    @property
    def predict_module_names(self) -> List[str]:
        return ["conv1", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]

    @property
    def final_feature_extractor(self) -> str:
        return "layer4"


resnet.ResNet = ResNet  # monkey patch


class DenseNet(FeatureExtractorMixin, densenet.DenseNet):
    @property
    def module_sequence(self) -> OrderedDict:
        seq = OrderedDict(self.features._modules.items())
        seq.update(
            OrderedDict(
                (
                    ("relu", nn.ReLU(inplace=True)),
                    ("avgpool", nn.AdaptiveAvgPool2d((1, 1))),
                    ("flatten", nn.Flatten(1)),
                    ("classifier", self.classifier),
                )
            )
        )
        return seq

    @property
    def predict_module_names(self) -> List[str]:
        return [*self.features._modules.keys()] + ["avgpool", "classifier"]

    @property
    def final_feature_extractor(self) -> str:
        return self.predict_module_names[-3]


densenet.DenseNet = DenseNet  # monkey patch


class VGG(FeatureExtractorMixin, vgg.VGG):
    @property
    def module_sequence(self) -> OrderedDict:
        seq = OrderedDict()
        c_cnt = m_cnt = r_cnt = b_cnt = 0
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                m_cnt += 1
                seq[f"maxpool{m_cnt}"] = layer
            elif isinstance(layer, nn.Conv2d):
                c_cnt += 1
                seq[f"conv_no_act{c_cnt}"] = layer
            elif isinstance(layer, nn.ReLU):
                r_cnt += 1
                seq[f"conv{r_cnt}"] = layer
            elif isinstance(layer, nn.BatchNorm2d):
                b_cnt += 1
                seq[f"bn{b_cnt}"] = layer
            else:
                raise RuntimeError(f"Unexpected layer type {type(layer)}")
        seq.update(
            OrderedDict(
                (
                    ("avgpool", self.avgpool),
                    ("flatten", nn.Flatten(1)),
                )
            )
        )
        seq.update(sequential_to_dict(self.classifier, prefix="classifier", last="classifier"))
        return seq

    @property
    def predict_module_names(self) -> List[str]:
        names = ["conv1"]
        m_cnt = 0
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                m_cnt += 1
                names.append(f"maxpool{m_cnt}")
        return names + ["avgpool", "classifier"]

    @property
    def final_feature_extractor(self) -> str:
        return self.predict_module_names[-3]


vgg.VGG = VGG  # monkey patch


class Inception3(FeatureExtractorMixin, inception.Inception3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.AuxLogits is not None:
            logger.warning("AuxLogits are not supported, ignoring.")

    @property
    def module_sequence(self) -> OrderedDict:
        return OrderedDict(
            (
                ("Conv2d_1a_3x3", self.Conv2d_1a_3x3),
                ("Conv2d_2a_3x3", self.Conv2d_2a_3x3),
                ("Conv2d_2b_3x3", self.Conv2d_2b_3x3),
                ("maxpool1", self.maxpool1),
                ("Conv2d_3b_1x1", self.Conv2d_3b_1x1),
                ("Conv2d_4a_3x3", self.Conv2d_4a_3x3),
                ("maxpool2", self.maxpool2),
                ("Mixed_5b", self.Mixed_5b),
                ("Mixed_5c", self.Mixed_5c),
                ("Mixed_5d", self.Mixed_5d),
                ("Mixed_6a", self.Mixed_6a),
                ("Mixed_6b", self.Mixed_6b),
                ("Mixed_6c", self.Mixed_6c),
                ("Mixed_6d", self.Mixed_6d),
                ("Mixed_6e", self.Mixed_6e),
                ("Mixed_7a", self.Mixed_7a),
                ("Mixed_7b", self.Mixed_7b),
                ("Mixed_7c", self.Mixed_7c),
                ("avgpool", self.avgpool),
                ("flatten", nn.Flatten(1)),
                ("fc", self.fc),
            )
        )

    @property
    def predict_module_names(self) -> List[str]:
        return [
            "Conv2d_1a_3x3",
            "Conv2d_2a_3x3",
            "Conv2d_2b_3x3",
            "maxpool1",
            "Conv2d_3b_1x1",
            "Conv2d_4a_3x3",
            "maxpool2",
            "Mixed_5b",
            "Mixed_5c",
            "Mixed_5d",
            "Mixed_6a",
            "Mixed_6b",
            "Mixed_6c",
            "Mixed_6d",
            "Mixed_6e",
            "Mixed_7c",
            "avgpool",
            "fc",
        ]

    @property
    def final_feature_extractor(self) -> str:
        return "Mixed_7c"


inception.Inception3 = Inception3  # monkey patch


class SqueezeNet(FeatureExtractorMixin, squeezenet.SqueezeNet):
    def __init__(self, version: str = "1_0", **kwargs) -> None:
        super().__init__(version=version, **kwargs)
        if version == "1_0":
            self.checkpoints = OrderedDict(
                ((0, "conv1"), (2, "maxpool1"), (6, "maxpool2"), (11, "maxpool3"), (12, "features"))
            )
        elif version == "1_1":
            self.checkpoints = OrderedDict(
                ((0, "conv1"), (2, "maxpool1"), (5, "maxpool2"), (8, "maxpool3"), (12, "features"))
            )
        else:
            raise NotImplementedError(version)
        self.classifier_checkpoints = OrderedDict(((1, "final_conv"),))

    @property
    def module_sequence(self) -> OrderedDict:
        seq = OrderedDict()
        feat_seq = sequential_to_dict(self.features)
        for i, (key, val) in enumerate(feat_seq.items()):
            if i in self.checkpoints:
                seq[self.checkpoints[i]] = val
            else:
                seq[key] = val
        classifier_seq = sequential_to_dict(self.classifier, prefix="classifier")
        for i, (key, val) in enumerate(classifier_seq.items()):
            if i in self.classifier_checkpoints:
                seq[self.classifier_checkpoints[i]] = val
            else:
                seq[key] = val
        seq["flatten"] = nn.Flatten(1)
        return seq

    @property
    def predict_module_names(self) -> List[str]:
        return [*self.checkpoints.values(), *self.classifier_checkpoints.values()]

    @property
    def final_feature_extractor(self) -> str:
        return "features"


squeezenet.SqueezeNet = SqueezeNet

all_model_names = {name for name in dir(models) if name[0].islower()}
supported_models = {}
for _module in (resnet, densenet, vgg, inception, squeezenet):
    for _name in all_model_names & {*dir(_module)}:
        supported_models[_name] = getattr(_module, _name)


def get_feature_extractor_base(
    name: str,
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> FeatureExtractorMixin:
    if name not in supported_models:
        raise ValueError(f'Unknown model "{name}". Valid names: ' f"{supported_models}")
    model_cls = supported_models[name]
    if version.parse("0.13.a") <= version.parse(torchvision.__version__):
        # post-0.13
        pretrained_kw = {"weights": "DEFAULT" if pretrained else None}
    else:
        # pre-0.13
        pretrained_kw = {"pretrained": pretrained}
    model = model_cls(progress=progress, **pretrained_kw, **kwargs)

    return model


def get_feature_extractor(
    name: str,
    pretrained: bool = False,
    progress: bool = True,
    last_module_name: Optional[Union[Iterable[str], str]] = None,
    append_last_layer: bool = False,
    **kwargs: Any,
) -> FeatureExtractorHackjob:
    base_model = get_feature_extractor_base(name, pretrained, progress, **kwargs)
    if append_last_layer:
        if last_module_name is None:
            last_module_name = [base_model.final_feature_extractor]
        elif isinstance(last_module_name, str):
            last_module_name = [last_module_name]
        last_module_name = [*last_module_name, base_model.last_layer]

    return FeatureExtractorHackjob(
        base_model,
        last_module_name=last_module_name,
    )
