# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) PyTorch Contributors 2022
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: BSD-3-Clause

# Code largely based on PyTorch https://github.com/pytorch/pytorch

import copy
import math
import os
import os.path as osp
import pickle
import sys
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union, cast

import numpy as np
from filelock import FileLock

import pixpnet
import pixpnet.symbolic.index_layers as nn
from pixpnet.symbolic.misc import _make_divisible, _overwrite_named_param, sym_scope
from pixpnet.utils import get_logger

logger = get_logger(__name__)

major, minor = sys.version_info[:2]
if major > 3 or (major == 3 and minor >= 9):
    OrderedDict_T = OrderedDict
else:
    OrderedDict_T = Dict

unique_syms = nn.unique_syms
Tensor = nn.Tensor

ROOT_DIR = osp.dirname(osp.dirname(osp.realpath(pixpnet.__file__)))
CACHE_DIR = osp.join(ROOT_DIR, "rf_cache")


def _get_cache_dir(model_name, height, width, num_classes, insert_at):
    insert_at_args = (f"insert_at_{insert_at}",) if insert_at else ()
    return osp.join(CACHE_DIR, model_name, f"{height}x{width}", f"{num_classes}_classes", *insert_at_args)


def _get_cache_lockfile(model_name, height, width, num_classes, insert_at):
    os.makedirs(CACHE_DIR, exist_ok=True)
    insert_at_str = f"__insert_at_{insert_at}" if insert_at else ""
    return osp.join(CACHE_DIR, f".{model_name}__{height}x{width}__{num_classes}_classes{insert_at_str}" f".lock")


def check_cache(model_name, height, width, num_classes=1, insert_at=None):
    save_dir = _get_cache_dir(model_name, height, width, num_classes, insert_at)
    save_path = osp.join(save_dir, "rf_data.pkl")
    if os.path.isfile(save_path):
        return save_path


def _serialize_ndarray(arr: np.ndarray):
    return {
        "shape": arr.shape,
        "data": [v.serialize() for v in arr.flat],
    }


def _deserialize_ndarray(data):
    return np.asarray([nn.HypercubeCollection.deserialize(arr_indices) for arr_indices in data["data"]]).reshape(
        data["shape"]
    )


def write_cache(out, intermediates, model_name, height, width, num_classes, insert_at):
    save_dir = _get_cache_dir(model_name, height, width, num_classes, insert_at)
    save_path = osp.join(save_dir, "rf_data.pkl")
    if os.path.isfile(save_path):
        logger.warning(f'Will overwrite "{save_path}" which already exists')
    else:
        os.makedirs(save_dir, exist_ok=True)

    write_data = {
        "out": out,
        "intermediates": [(k, v) for k, v in intermediates.items()],
    }
    with open(save_path, "wb") as fp:
        pickle.dump(write_data, fp)
        fp.flush()


def load_cache(model_name, height, width, num_classes=1, insert_at=None):
    save_dir = _get_cache_dir(model_name, height, width, num_classes, insert_at)
    save_path = osp.join(save_dir, "rf_data.pkl")
    with open(save_path, "rb") as fp:
        sys.modules["ngn"] = pixpnet  # legacy naming
        data = pickle.load(fp)
    logger.info(f'Reusing cached data "{save_path}"')
    out = data["out"]
    intermediates = OrderedDict(((k, v) for k, v in data["intermediates"]))
    return out, intermediates


def compute_rf_data(model_name, height, width, num_classes=1, insert_at=None):
    name_is_name = isinstance(model_name, str)

    lock_path = _get_cache_lockfile(model_name, height, width, num_classes, insert_at)
    with FileLock(lock_path):
        if name_is_name and check_cache(model_name, height, width, num_classes, insert_at):
            try:
                out, intermediates = load_cache(model_name, height, width, num_classes, insert_at)
            except pickle.UnpicklingError:
                logger.warning("UnpicklingError when loading rf data! " "Recomputing...")
            else:
                return out, intermediates
        # It is not in the cache at this point.
        if name_is_name:
            try:
                sym_model_cls = globals()[model_name]
            except KeyError:
                raise ValueError(f'Invalid name "{model_name}". Valid: ' f"{[*globals().keys()]}")
        else:
            sym_model_cls = model_name
        img_shape = (height, width)

        with unique_syms() as ctx:
            x = ctx.Tensor(shape=(1, 1, *img_shape), name="x")

        sym_model = sym_model_cls(num_classes=num_classes)

        if insert_at:
            _, rf_data_from_x = compute_rf_data(model_name, height, width, num_classes)
            shape_at_insert_layer = rf_data_from_x[insert_at].shape
            with unique_syms() as ctx:
                intermediate_x = ctx.Tensor(shape=shape_at_insert_layer, name="intermediate_x")
            out, intermediates = sym_model(x, intermediate_x=intermediate_x, insert_at=insert_at)
        else:
            out, intermediates = sym_model(x)
        if name_is_name:
            write_cache(out, intermediates, model_name, height, width, num_classes, insert_at)
        return out, intermediates


class StochasticDepth(nn.Module):
    """
    See :func:`stochastic_depth`.
    """

    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        return input

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"
        return s


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        with sym_scope("conv1"):
            self.conv1 = conv3x3(inplanes, planes, stride)
        with sym_scope("relu"):
            self.relu = nn.ReLU()
        with sym_scope("conv2"):
            self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3
    # convolution(self.conv2) while original implementation places the stride at
    # the first 1x1 convolution(self.conv1) according to "Deep residual learning
    # for image recognition"https://arxiv.org/abs/1512.03385. This variant is
    # also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    # 4 normally but not needed in symbolic version (more than 1 channel is
    # redundant)
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        with sym_scope("conv1"):
            self.conv1 = conv1x1(inplanes, width)
        with sym_scope("conv2"):
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        with sym_scope("conv3"):
            self.conv3 = conv1x1(width, planes * self.expansion)
        with sym_scope("relu"):
            self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        with sym_scope("conv1"):  # 3 --> 1 channel
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        with sym_scope("relu"):
            self.relu = nn.ReLU()
        with sym_scope("maxpool"):
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        with sym_scope("layer1"):
            self.layer1 = self._make_layer(block, 64, layers[0])  # 64
        with sym_scope("layer2"):
            self.layer2 = self._make_layer(
                block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]  # 128
            )
        with sym_scope("layer3"):
            self.layer3 = self._make_layer(
                block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]  # 256
            )
        with sym_scope("layer4"):
            self.layer4 = self._make_layer(
                block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]  # 512
            )
        with sym_scope("avgpool"):
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        with sym_scope("fc"):
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # 512

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            with sym_scope("downsample"):
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                )

        with sym_scope("block0"):
            layers = [
                block(
                    self.inplanes,
                    planes,
                    stride,
                    downsample,
                    self.groups,
                    self.base_width,
                    previous_dilation,
                    norm_layer,
                )
            ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            with sym_scope(f"block_{i}"):
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        groups=self.groups,
                        base_width=self.base_width,
                        dilation=self.dilation,
                        norm_layer=norm_layer,
                    )
                )

        return nn.Sequential(*layers)

    def _forward_impl(self, x, intermediate_x: Optional[nn.Tensor] = None, insert_at: Optional[str] = None):
        assert not ((insert_at is None) ^ (intermediate_x is None))
        intermediates = OrderedDict()
        x = self.conv1(x)
        x = self.relu(x)
        intermediates["conv1"] = x
        if insert_at == "conv1":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        x = self.maxpool(x)
        intermediates["maxpool"] = x
        if insert_at == "maxpool":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x

        x = self.layer1(x)
        intermediates["layer1"] = x
        if insert_at == "layer1":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        x = self.layer2(x)
        intermediates["layer2"] = x
        if insert_at == "layer2":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        x = self.layer3(x)
        intermediates["layer3"] = x
        if insert_at == "layer3":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        x = self.layer4(x)
        intermediates["layer4"] = x
        if insert_at == "layer4":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x

        x = self.avgpool(x)
        intermediates["avgpool"] = x
        if insert_at == "avgpool":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        x = nn.flatten(x, 1)
        x = self.fc(x)
        intermediates["fc"] = x
        if insert_at == "fc":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x

        return x, intermediates

    def forward(self, x, intermediate_x: Optional[nn.Tensor] = None, insert_at: Optional[str] = None):
        return self._forward_impl(x, intermediate_x, insert_at)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition
    <https://arxiv.org/pdf/1512.03385.pdf>`__.
    Args:
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    """
    with sym_scope("resnet18"):
        return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs: Any) -> ResNet:
    """ResNet-34 from `Deep Residual Learning for Image Recognition
    <https://arxiv.org/pdf/1512.03385.pdf>`__.
    Args:
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    """
    with sym_scope("resnet34"):
        return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition
    <https://arxiv.org/pdf/1512.03385.pdf>`__.
    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the
       second 3x3 convolution while the original paper places it to the first
       1x1 convolution. This variant improves the accuracy and is known as
       `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.
    Args:
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    """
    with sym_scope("resnet50"):
        return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs: Any) -> ResNet:
    """ResNet-101 from `Deep Residual Learning for Image Recognition
    <https://arxiv.org/pdf/1512.03385.pdf>`__.
    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the
       second 3x3 convolution while the original paper places it to the first
       1x1 convolution. This variant improves the accuracy and is known as
       `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.
    Args:
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    """
    with sym_scope("resnet101"):
        return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs: Any) -> ResNet:
    """ResNet-152 from `Deep Residual Learning for Image Recognition
    <https://arxiv.org/pdf/1512.03385.pdf>`__.
    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the
       second 3x3 convolution while the original paper places it to the first
       1x1 convolution. This variant improves the accuracy and is known as
       `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.
    Args:
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    """
    with sym_scope("resnet152"):
        return _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnext50_32x4d(**kwargs: Any) -> ResNet:
    """ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks
    <https://arxiv.org/abs/1611.05431>`_.
    Args:
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    """
    _overwrite_named_param(kwargs, "groups", 32)
    _overwrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnext101_32x8d(**kwargs: Any) -> ResNet:
    """ResNeXt-101 32x8d model from
    `Aggregated Residual Transformation for Deep Neural Networks
    <https://arxiv.org/abs/1611.05431>`_.
    Args:
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    """
    _overwrite_named_param(kwargs, "groups", 32)
    _overwrite_named_param(kwargs, "width_per_group", 8)
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnext101_64x4d(**kwargs: Any) -> ResNet:
    """ResNeXt-101 64x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks
    <https://arxiv.org/abs/1611.05431>`_.
    Args:
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    """
    _overwrite_named_param(kwargs, "groups", 64)
    _overwrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def wide_resnet50_2(**kwargs: Any) -> ResNet:
    """Wide ResNet-50-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    """
    _overwrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def wide_resnet101_2(**kwargs: Any) -> ResNet:
    """Wide ResNet-101-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-101 has 2048-512-2048
    channels, and in Wide ResNet-101-2 has 2048-1024-2048.
    Args:
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    """
    _overwrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


class VGG(nn.Module):
    def __init__(self, features, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5) -> None:
        super().__init__()
        self.features, self.cfg = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(1 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: nn.Tensor, intermediate_x: Optional[nn.Tensor] = None, insert_at: Optional[str] = None):
        assert not ((insert_at is None) ^ (intermediate_x is None))
        intermediates = OrderedDict()
        m_cnt = 0
        for i, (layer, typ) in enumerate(zip(self.features, self.cfg)):
            x = layer(x)
            if typ == "M":
                m_cnt += 1
                intermediates[f"maxpool{m_cnt}"] = x
                if insert_at == f"maxpool{m_cnt}":
                    intermediates = OrderedDict()
                    intermediates[insert_at] = intermediate_x
                    x = intermediate_x
            elif i == 0:
                intermediates["conv1"] = x
                if insert_at == "conv1":
                    intermediates = OrderedDict()
                    intermediates[insert_at] = intermediate_x
                    x = intermediate_x
        x = self.avgpool(x)
        intermediates["avgpool"] = x
        if insert_at == "avgpool":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        x = nn.flatten(x, 1)
        x = self.classifier(x)
        intermediates["classifier"] = x
        if insert_at == "classifier":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        return x, intermediates


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False):
    layers: List[nn.Module] = []
    in_channels = 1
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            v = 1  # always one channel b
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [nn.Sequential(conv2d, nn.ReLU())]
            in_channels = v
    return layers, cfg


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11", "A", False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11_bn", "A", True, pretrained, progress, **kwargs)


def vgg13(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13", "B", False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13_bn", "B", True, pretrained, progress, **kwargs)


def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16", "D", False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16_bn", "D", True, pretrained, progress, **kwargs)


def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19", "E", False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19_bn", "E", True, pretrained, progress, **kwargs)


class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU()
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU()
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU()

    def forward(self, x: nn.Tensor) -> nn.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return nn.cat([self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1)


class SqueezeNet(nn.Module):
    def __init__(self, version: str = "1_0", num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        if version == "1_0":
            self.features = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=7, stride=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(1, 1, 1, 1),
                Fire(2, 1, 1, 1),
                Fire(2, 1, 1, 1),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(2, 1, 1, 1),
                Fire(2, 1, 1, 1),
                Fire(2, 1, 1, 1),
                Fire(2, 1, 1, 1),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(2, 1, 1, 1),
            )
            self.checkpoints = {0: "conv1", 2: "maxpool1", 6: "maxpool2", 11: "maxpool3"}
        elif version == "1_1":
            self.features = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(1, 1, 1, 1),
                Fire(2, 1, 1, 1),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(2, 1, 1, 1),
                Fire(2, 1, 1, 1),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(2, 1, 1, 1),
                Fire(2, 1, 1, 1),
                Fire(2, 1, 1, 1),
                Fire(2, 1, 1, 1),
            )
            self.checkpoints = {0: "conv1", 2: "maxpool1", 5: "maxpool2", 8: "maxpool3"}
        else:
            raise ValueError(f"Unsupported SqueezeNet version {version}: " f"1_0 or 1_1 expected")

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(2, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(final_conv, nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier_checkpoints = {0: "final_conv", 2: "avgpool"}

    def forward(self, x: nn.Tensor, intermediate_x: Optional[nn.Tensor] = None, insert_at: Optional[str] = None):
        assert not ((insert_at is None) ^ (intermediate_x is None))
        intermediates = OrderedDict()
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.checkpoints:
                intermediates[self.checkpoints[i]] = x
                if insert_at == self.checkpoints[i]:
                    intermediates = OrderedDict()
                    intermediates[insert_at] = intermediate_x
                    x = intermediate_x
        intermediates["features"] = x
        if insert_at == "features":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if i in self.classifier_checkpoints:
                intermediates[self.classifier_checkpoints[i]] = x
                if insert_at == self.classifier_checkpoints[i]:
                    intermediates = OrderedDict()
                    intermediates[insert_at] = intermediate_x
                    x = intermediate_x
        return nn.flatten(x, 1), intermediates


def _squeezenet(version: str, pretrained: bool, progress: bool, **kwargs: Any) -> SqueezeNet:
    model = SqueezeNet(version, **kwargs)
    return model


def squeezenet1_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SqueezeNet:
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    The required minimum input size of the model is 21x21.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet("1_0", pretrained, progress, **kwargs)


def squeezenet1_1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SqueezeNet:
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    The required minimum input size of the model is 17x17.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet("1_1", pretrained, progress, **kwargs)


class Inception3(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = False,
        transform_input: bool = False,
        inception_blocks: Optional[List[Callable[..., nn.Module]]] = None,
        init_weights: Optional[bool] = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if inception_blocks is None:
            inception_blocks = [BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, InceptionAux]
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(1, 1, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(1, 1, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(1, 1, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(1, 1, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(1, 1, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(1, pool_features=1)
        self.Mixed_5c = inception_a(4, pool_features=1)
        self.Mixed_5d = inception_a(4, pool_features=1)
        self.Mixed_6a = inception_b(4)
        self.Mixed_6b = inception_c(6, channels_7x7=1)
        self.Mixed_6c = inception_c(4, channels_7x7=1)
        self.Mixed_6d = inception_c(4, channels_7x7=1)
        self.Mixed_6e = inception_c(4, channels_7x7=1)
        self.AuxLogits: Optional[nn.Module] = None
        if aux_logits:
            self.AuxLogits = inception_aux(1, num_classes)
        self.Mixed_7a = inception_d(4)
        self.Mixed_7b = inception_e(6)
        self.Mixed_7c = inception_e(6)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(6, num_classes)

    def _transform_input(self, x: nn.Tensor) -> nn.Tensor:
        if self.transform_input:
            x_ch0 = nn.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = nn.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = nn.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = nn.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: nn.Tensor, intermediate_x: Optional[nn.Tensor] = None, insert_at: Optional[str] = None):
        assert not ((insert_at is None) ^ (intermediate_x is None))
        intermediates = OrderedDict()
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        intermediates["Conv2d_1a_3x3"] = x
        if insert_at == "Conv2d_1a_3x3":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        intermediates["Conv2d_2a_3x3"] = x
        if insert_at == "Conv2d_2a_3x3":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        intermediates["Conv2d_2b_3x3"] = x
        if insert_at == "Conv2d_2b_3x3":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        intermediates["maxpool1"] = x
        if insert_at == "maxpool1":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        intermediates["Conv2d_3b_1x1"] = x
        if insert_at == "Conv2d_3b_1x1":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        intermediates["Conv2d_4a_3x3"] = x
        if insert_at == "Conv2d_4a_3x3":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        intermediates["maxpool2"] = x
        if insert_at == "maxpool2":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        intermediates["Mixed_5b"] = x
        if insert_at == "Mixed_5b":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        intermediates["Mixed_5c"] = x
        if insert_at == "Mixed_5c":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        intermediates["Mixed_5d"] = x
        if insert_at == "Mixed_5d":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        intermediates["Mixed_6a"] = x
        if insert_at == "Mixed_6a":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        intermediates["Mixed_6b"] = x
        if insert_at == "Mixed_6b":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        intermediates["Mixed_6c"] = x
        if insert_at == "Mixed_6c":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        intermediates["Mixed_6d"] = x
        if insert_at == "Mixed_6d":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        intermediates["Mixed_6e"] = x
        if insert_at == "Mixed_6e":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        # N x 768 x 17 x 17
        aux: Optional[nn.Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        intermediates["Mixed_7c"] = x
        if insert_at == "Mixed_7c":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        intermediates["avgpool"] = x
        if insert_at == "avgpool":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        # N x 2048 x 1 x 1
        x = nn.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        intermediates["fc"] = x
        if insert_at == "fc":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        # N x 1000 (num_classes)
        return x, aux, intermediates

    def eager_outputs(self, x: nn.Tensor, aux: Optional[nn.Tensor]):
        return x  # type: ignore[return-value]

    def forward(self, x: nn.Tensor, intermediate_x: Optional[nn.Tensor] = None, insert_at: Optional[str] = None):
        x = self._transform_input(x)
        x, aux, intermediates = self._forward(x, intermediate_x, insert_at)
        return self.eager_outputs(x, aux), intermediates


class InceptionA(nn.Module):
    def __init__(
        self, in_channels: int, pool_features: int, conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 1, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 1, kernel_size=1)
        self.branch5x5_2 = conv_block(1, 1, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 1, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(1, 1, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(1, 1, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def _forward(self, x: nn.Tensor) -> List[nn.Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.avgpool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: nn.Tensor) -> nn.Tensor:
        outputs = self._forward(x)
        return nn.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 1, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 1, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(1, 1, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(1, 1, kernel_size=3, stride=2)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def _forward(self, x: nn.Tensor) -> List[nn.Tensor]:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.maxpool(x)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: nn.Tensor) -> nn.Tensor:
        outputs = self._forward(x)
        return nn.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(
        self, in_channels: int, channels_7x7: int, conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 1, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 1, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 1, kernel_size=(1, 7), padding=(0, 3))
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool = conv_block(in_channels, 1, kernel_size=1)

    def _forward(self, x: nn.Tensor) -> List[nn.Tensor]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = self.avgpool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x: nn.Tensor) -> nn.Tensor:
        outputs = self._forward(x)
        return nn.cat(outputs, 1)


class InceptionD(nn.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 1, kernel_size=1)
        self.branch3x3_2 = conv_block(1, 1, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 1, kernel_size=1)
        self.branch7x7x3_2 = conv_block(1, 1, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(1, 1, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(1, 1, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def _forward(self, x: nn.Tensor) -> List[nn.Tensor]:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = self.maxpool(x)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x: nn.Tensor) -> nn.Tensor:
        outputs = self._forward(x)
        return nn.cat(outputs, 1)


class InceptionE(nn.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 1, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 1, kernel_size=1)
        self.branch3x3_2a = conv_block(1, 1, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(1, 1, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 1, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(1, 1, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(1, 1, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(1, 1, kernel_size=(3, 1), padding=(1, 0))
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool = conv_block(in_channels, 1, kernel_size=1)

    def _forward(self, x: nn.Tensor) -> List[nn.Tensor]:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = nn.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = nn.cat(branch3x3dbl, 1)

        branch_pool = self.avgpool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: nn.Tensor) -> nn.Tensor:
        outputs = self._forward(x)
        return nn.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(
        self, in_channels: int, num_classes: int, conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 1, kernel_size=1)
        self.conv1 = conv_block(1, 1, kernel_size=5)
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.fc = nn.Linear(1, num_classes)
        self.fc.stddev = 0.001  # type: ignore[assignment]
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)

    def forward(self, x: nn.Tensor) -> nn.Tensor:
        # N x 768 x 17 x 17
        x = self.avgpool(x)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = self.adaptive_avgpool(x)
        # N x 768 x 1 x 1
        x = nn.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x: nn.Tensor) -> nn.Tensor:
        x = self.conv(x)
        return self.relu(x)


def inception_v3(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Inception3:
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision"
    <http://arxiv.org/abs/1512.00567>`_.
    The required minimum input size of the model is 75x75.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects
        nn.Tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to
            stderr
        aux_logits (bool): If True, add an auxiliary branch that can improve
            training.
            Default: *True*
        transform_input (bool): If True, preprocesses the input according to the
            method with which it was trained on ImageNet. Default: True if
            ``pretrained=True``, else False.
    """
    return Inception3(**kwargs)


class _DenseLayer(nn.Module):
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False
    ) -> None:
        super().__init__()
        self.relu1: nn.ReLU
        self.add_module("relu1", nn.ReLU())
        self.conv1: nn.Conv2d
        self.add_module(
            "conv1", nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        )
        self.relu2: nn.ReLU
        self.add_module("relu2", nn.ReLU())
        self.conv2: nn.Conv2d
        self.add_module(
            "conv2", nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        concated_features = nn.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(concated_features))  # noqa: T484
        return bottleneck_output

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if not isinstance(input, list):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(bottleneck_output))
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return nn.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.add_module("relu", nn.ReLU())
        self.add_module("conv", nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
        memory_efficient: bool = False,
    ) -> None:
        growth_rate = 1
        num_init_features = 1
        bn_size = 1

        super().__init__()

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("relu0", nn.ReLU()),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=max(num_features // 2, 1))
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = max(num_features // 2, 1)

        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x, intermediate_x: Optional[nn.Tensor] = None, insert_at: Optional[str] = None):
        assert not ((insert_at is None) ^ (intermediate_x is None))
        intermediates = OrderedDict()
        for name, layer in self.features.items():
            x = layer(x)
            intermediates[name] = x
            if insert_at == name:
                intermediates = OrderedDict()
                intermediates[insert_at] = intermediate_x
                x = intermediate_x
        features = x
        out = self.relu(features)
        out = self.avgpool(out)
        intermediates["avgpool"] = out
        if insert_at == "avgpool":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            out = intermediate_x
        out = nn.flatten(out, 1)
        out = self.classifier(out)
        intermediates["classifier"] = out
        if insert_at == "classifier":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            out = intermediate_x
        return out, intermediates


def _densenet(
    arch: str,
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> DenseNet:
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    return model


def densenet121(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet("densenet121", 32, (6, 12, 24, 16), 64, pretrained, progress, **kwargs)


def densenet161(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet("densenet161", 48, (6, 12, 36, 24), 96, pretrained, progress, **kwargs)


def densenet169(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet("densenet169", 32, (6, 12, 32, 32), 64, pretrained, progress, **kwargs)


def densenet201(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet("densenet201", 32, (6, 12, 48, 32), 64, pretrained, progress, **kwargs)


class ConvNormActivation(nn.Sequential):
    """
    Configurable block used for Convolution-Normalzation-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-
        Normalzation-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides
        of the input. Default: None, in wich case it will calculated as
        ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input
        channels to output channels. Default: 1
        norm_layer (Callable[..., nn.Module], optional): Norm layer that will be
        stacked on top of the convolutiuon layer. If ``None`` this layer wont be
        used. Default: ``nn.BatchNorm2d``
        activation_layer (Callable[..., nn.Module], optinal): Activation
            function which will be stacked on top of the normalization layer
            (if not None), otherwise on top of the conv layer. If ``None`` this
            layer wont be used. Default: ``nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally
            do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By
            default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:
        norm_layer = None
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels


class SqueezeExcitation(nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from
    https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta``
    and ``sigma`` in in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., nn.Module], optional): ``delta`` activation.
            Default: ``nn.ReLU``
        scale_activation (Callable[..., nn.Module]): ``sigma`` activation.
            Default: ``nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., nn.Module] = nn.ReLU,
        scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input):
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input):
        scale = self._scale(input)
        return scale * input


class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float,
        depth_mult: float,
    ) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        width_mult = 1
        input_channels = out_channels = 1
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"expand_ratio={self.expand_ratio}"
            f", kernel={self.kernel}"
            f", stride={self.stride}"
            f", input_channels={self.input_channels}"
            f", out_channels={self.out_channels}"
            f", num_layers={self.num_layers}"
            f")"
        )
        return s

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 1, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            ConvNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(
            ConvNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels

    def forward(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class EfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[MBConvConfig],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        """
        EfficientNet main class

        Args:
            inverted_residual_setting (List[MBConvConfig]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying
                inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying
                the normalization layer to use
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all(isinstance(s, MBConvConfig) for s in inverted_residual_setting)
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if block is None:
            block = MBConv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        norm_layer = None

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                1, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the
                # stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 1 * lastconv_input_channels
        layers.append(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, num_classes),
        )

    def _forward_impl(self, x, intermediate_x: Optional[nn.Tensor] = None, insert_at: Optional[str] = None):
        assert not ((insert_at is None) ^ (intermediate_x is None))
        intermediates = OrderedDict()
        for name, layer in self.features.items():
            x = layer(x)
            intermediates[name] = x
            if insert_at == name:
                intermediates = OrderedDict()
                intermediates[insert_at] = intermediate_x
                x = intermediate_x

        x = self.avgpool(x)
        intermediates["avgpool"] = x
        if insert_at == "avgpool":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        x = nn.flatten(x, 1)

        x = self.classifier(x)
        intermediates["classifier"] = x
        if insert_at == "classifier":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x

        return x, intermediates

    def forward(self, x, intermediate_x: Optional[nn.Tensor] = None, insert_at: Optional[str] = None):
        return self._forward_impl(x, intermediate_x, insert_at)


def _efficientnet(
    arch: str,
    width_mult: float,
    depth_mult: float,
    dropout: float,
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> EfficientNet:
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    model = EfficientNet(inverted_residual_setting, dropout, **kwargs)
    return model


def efficientnet_b0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B0 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b0", 1.0, 1.0, 0.2, pretrained, progress, **kwargs)


def efficientnet_b1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B1 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b1", 1.0, 1.1, 0.2, pretrained, progress, **kwargs)


def efficientnet_b2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B2 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b2", 1.1, 1.2, 0.3, pretrained, progress, **kwargs)


def efficientnet_b3(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B3 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b3", 1.2, 1.4, 0.3, pretrained, progress, **kwargs)


def efficientnet_b4(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B4 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b4", 1.4, 1.8, 0.4, pretrained, progress, **kwargs)


def efficientnet_b5(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B5 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet(
        "efficientnet_b5",
        1.6,
        2.2,
        0.4,
        pretrained,
        progress,
        **kwargs,
    )


def efficientnet_b6(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B6 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet(
        "efficientnet_b6",
        1.8,
        2.6,
        0.5,
        pretrained,
        progress,
        **kwargs,
    )


def efficientnet_b7(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B7 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet(
        "efficientnet_b7",
        2.0,
        3.1,
        0.5,
        pretrained,
        progress,
        **kwargs,
    )


class SimpleStemIN(ConvNormActivation):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__(
            width_in, width_out, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=activation_layer
        )


class BottleneckTransform(nn.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
    ) -> None:
        layers: OrderedDict_T[str, nn.Module] = OrderedDict()
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width

        layers["a"] = ConvNormActivation(
            width_in, w_b, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=activation_layer
        )
        layers["b"] = ConvNormActivation(
            w_b, w_b, kernel_size=3, stride=stride, groups=g, norm_layer=norm_layer, activation_layer=activation_layer
        )

        if se_ratio:
            # The SE reduction ratio is defined with respect to the
            # beginning of the block
            width_se_out = int(round(se_ratio * width_in))
            layers["se"] = SqueezeExcitation(
                input_channels=w_b,
                squeeze_channels=width_se_out,
                activation=activation_layer,
            )

        layers["c"] = ConvNormActivation(
            w_b, width_out, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=None
        )
        super().__init__(layers)


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int = 1,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()

        # Use skip connection with projection if shape changes
        self.proj = None
        should_proj = (width_in != width_out) or (stride != 1)
        if should_proj:
            self.proj = ConvNormActivation(
                width_in, width_out, kernel_size=1, stride=stride, norm_layer=norm_layer, activation_layer=None
            )
        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            norm_layer,
            activation_layer,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        )
        self.activation = activation_layer(inplace=True)

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class AnyStage(nn.Sequential):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        block_constructor: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float] = None,
        stage_index: int = 0,
    ) -> None:
        super().__init__()

        for i in range(depth):
            block = block_constructor(
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1,
                norm_layer,
                activation_layer,
                group_width,
                bottleneck_multiplier,
                se_ratio,
            )

            self.add_module(f"block{stage_index}-{i}", block)


class BlockParams:
    def __init__(
        self,
        depths: List[int],
        widths: List[int],
        group_widths: List[int],
        bottleneck_multipliers: List[float],
        strides: List[int],
        se_ratio: Optional[float] = None,
    ) -> None:
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def from_init_params(
        cls,
        depth: int,
        w_0: int,
        w_a: float,
        w_m: float,
        group_width: int,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
        **kwargs: Any,
    ) -> "BlockParams":
        """
        Programatically compute all the per-block settings,
        given the RegNet parameters.

        The first step is to compute the quantized linear block parameters,
        in log space. Key parameters are:
        - `w_a` is the width progression slope
        - `w_0` is the initial width
        - `w_m` is the width stepping in the log space

        In other terms
        `log(block_width) = log(w_0) + w_m * block_capacity`,
        with `bock_capacity` ramping up following the w_0 and w_a params.
        This block width is finally quantized to multiples of 8.

        The second step is to compute the parameters per stage,
        taking into account the skip connection and the final 1x1 convolutions.
        We use the fact that the output width is constant within a stage.
        """

        QUANT = 8
        STRIDE = 2

        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError("Invalid RegNet settings")
        # Compute the block widths. Each stage has one unique block width
        import torch

        widths_cont = torch.arange(depth) * w_a + w_0
        block_capacity = torch.round(torch.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (torch.round(torch.divide(w_0 * torch.pow(w_m, block_capacity), QUANT)) * QUANT).int().tolist()
        num_stages = len(set(block_widths))

        # Convert to per stage parameters
        split_helper = zip(
            block_widths + [0],
            [0] + block_widths,
            block_widths + [0],
            [0] + block_widths,
        )
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = torch.diff(torch.tensor([d for d, t in enumerate(splits) if t])).int().tolist()

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages

        # Adjust the compatibility of stage widths and group widths
        stage_widths, group_widths = cls._adjust_widths_groups_compatibilty(
            stage_widths, bottleneck_multipliers, group_widths
        )

        return cls(
            depths=stage_depths,
            widths=stage_widths,
            group_widths=group_widths,
            bottleneck_multipliers=bottleneck_multipliers,
            strides=strides,
            se_ratio=se_ratio,
        )

    def _get_expanded_params(self):
        return zip(self.widths, self.strides, self.depths, self.group_widths, self.bottleneck_multipliers)

    @staticmethod
    def _adjust_widths_groups_compatibilty(
        stage_widths: List[int], bottleneck_ratios: List[float], group_widths: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Adjusts the compatibility of widths and groups,
        depending on the bottleneck ratio.
        """
        # Compute all widths for the current settings
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]

        # Compute the adjusted widths so that stage and group widths fit
        ws_bot = [_make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return stage_widths, group_widths_min


class RegNet(nn.Module):
    def __init__(
        self,
        block_params: BlockParams,
        num_classes: int = 1000,
        stem_width: int = 32,
        stem_type: Optional[Callable[..., nn.Module]] = None,
        block_type: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = None
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = nn.ReLU

        # Ad hoc stem
        self.stem = stem_type(
            1,  # width_in
            stem_width,
            norm_layer,
            activation,
        )

        current_width = stem_width

        blocks = []
        for i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(block_params._get_expanded_params()):
            blocks.append(
                (
                    f"block{i + 1}",
                    AnyStage(
                        current_width,
                        width_out,
                        stride,
                        depth,
                        block_type,
                        norm_layer,
                        activation,
                        group_width,
                        bottleneck_multiplier,
                        block_params.se_ratio,
                        stage_index=i + 1,
                    ),
                )
            )

            current_width = width_out

        self.trunk_output = nn.Sequential(OrderedDict(blocks))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=current_width, out_features=num_classes)

    def forward(self, x, intermediate_x: Optional[nn.Tensor] = None, insert_at: Optional[str] = None):
        assert not ((insert_at is None) ^ (intermediate_x is None))
        intermediates = OrderedDict()
        x = self.stem(x)
        intermediates["stem"] = x
        if insert_at == "stem":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        for name, layer in self.trunk_output.items():
            x = layer(x)
            intermediates[name] = x
            if insert_at == name:
                intermediates = OrderedDict()
                intermediates[insert_at] = intermediate_x
                x = intermediate_x

        x = self.avgpool(x)
        intermediates["avgpool"] = x
        if insert_at == "avgpool":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        x = nn.flatten(x, start_dim=1)
        x = self.fc(x)
        intermediates["fc"] = x
        if insert_at == "fc":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x

        return x, intermediates


def _regnet(arch: str, block_params: BlockParams, pretrained: bool, progress: bool, **kwargs: Any) -> RegNet:
    norm_layer = None
    model = RegNet(block_params, norm_layer=norm_layer, **kwargs)
    return model


def regnet_y_400mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_400MF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25, **kwargs)
    return _regnet("regnet_y_400mf", params, pretrained, progress, **kwargs)


def regnet_y_800mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_800MF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=14, w_0=56, w_a=38.84, w_m=2.4, group_width=16, se_ratio=0.25, **kwargs)
    return _regnet("regnet_y_800mf", params, pretrained, progress, **kwargs)


def regnet_y_1_6gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_1.6GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=27, w_0=48, w_a=20.71, w_m=2.65, group_width=24, se_ratio=0.25, **kwargs
    )
    return _regnet("regnet_y_1_6gf", params, pretrained, progress, **kwargs)


def regnet_y_3_2gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_3.2GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=21, w_0=80, w_a=42.63, w_m=2.66, group_width=24, se_ratio=0.25, **kwargs
    )
    return _regnet("regnet_y_3_2gf", params, pretrained, progress, **kwargs)


def regnet_y_8gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_8GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=17, w_0=192, w_a=76.82, w_m=2.19, group_width=56, se_ratio=0.25, **kwargs
    )
    return _regnet("regnet_y_8gf", params, pretrained, progress, **kwargs)


def regnet_y_16gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_16GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=18, w_0=200, w_a=106.23, w_m=2.48, group_width=112, se_ratio=0.25, **kwargs
    )
    return _regnet("regnet_y_16gf", params, pretrained, progress, **kwargs)


def regnet_y_32gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_32GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=20, w_0=232, w_a=115.89, w_m=2.53, group_width=232, se_ratio=0.25, **kwargs
    )
    return _regnet("regnet_y_32gf", params, pretrained, progress, **kwargs)


def regnet_y_128gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_128GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.
    NOTE: Pretrained weights are not available for this model.
    """
    params = BlockParams.from_init_params(
        depth=27, w_0=456, w_a=160.83, w_m=2.52, group_width=264, se_ratio=0.25, **kwargs
    )
    return _regnet("regnet_y_128gf", params, pretrained, progress, **kwargs)


def regnet_x_400mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_400MF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=22, w_0=24, w_a=24.48, w_m=2.54, group_width=16, **kwargs)
    return _regnet("regnet_x_400mf", params, pretrained, progress, **kwargs)


def regnet_x_800mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_800MF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=16, w_0=56, w_a=35.73, w_m=2.28, group_width=16, **kwargs)
    return _regnet("regnet_x_800mf", params, pretrained, progress, **kwargs)


def regnet_x_1_6gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_1.6GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24, **kwargs)
    return _regnet("regnet_x_1_6gf", params, pretrained, progress, **kwargs)


def regnet_x_3_2gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_3.2GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=25, w_0=88, w_a=26.31, w_m=2.25, group_width=48, **kwargs)
    return _regnet("regnet_x_3_2gf", params, pretrained, progress, **kwargs)


def regnet_x_8gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_8GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=23, w_0=80, w_a=49.56, w_m=2.88, group_width=120, **kwargs)
    return _regnet("regnet_x_8gf", params, pretrained, progress, **kwargs)


def regnet_x_16gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_16GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=22, w_0=216, w_a=55.59, w_m=2.1, group_width=128, **kwargs)
    return _regnet("regnet_x_16gf", params, pretrained, progress, **kwargs)


def regnet_x_32gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_32GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=23, w_0=320, w_a=69.86, w_m=2.0, group_width=168, **kwargs)
    return _regnet("regnet_x_32gf", params, pretrained, progress, **kwargs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: nn.Tensor) -> nn.Tensor:
        return x


class Permute(nn.Module):
    def __init__(self, dims: List[int], identical_channels=False):
        super().__init__()
        self.dims = dims
        self.identical_channels = identical_channels

    def forward(self, x):
        x = x.transpose(self.dims)
        if self.identical_channels:
            x.identical_channels = True
        return x


class CNBlock(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2], identical_channels=True),
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: nn.Tensor) -> nn.Tensor:
        result = self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
        self,
        input_channels: int,
        out_channels: Optional[int],
        num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)


class ConvNeXt(nn.Module):
    def __init__(
        self,
        block_setting: List[CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                1,  # 3
                firstconv_output_channels,
                kernel_size=4,
                stride=4,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        )
        self.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels), nn.Flatten(1), nn.Linear(lastconv_output_channels, num_classes)
        )

    def _forward_impl(self, x: nn.Tensor, intermediate_x: Optional[nn.Tensor] = None, insert_at: Optional[str] = None):
        assert not ((insert_at is None) ^ (intermediate_x is None))
        intermediates = OrderedDict()
        for name, layer in self.features.items():
            x = layer(x)
            intermediates[name] = x
            if insert_at == name:
                intermediates = OrderedDict()
                intermediates[insert_at] = intermediate_x
                x = intermediate_x
        x = self.avgpool(x)
        intermediates["avgpool"] = x
        if insert_at == "avgpool":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        x = self.classifier(x)
        intermediates["classifier"] = x
        if insert_at == "classifier":
            intermediates = OrderedDict()
            intermediates[insert_at] = intermediate_x
            x = intermediate_x
        return x, intermediates

    def forward(self, x: nn.Tensor, intermediate_x: Optional[nn.Tensor] = None, insert_at: Optional[str] = None):
        assert not ((insert_at is None) ^ (intermediate_x is None))
        return self._forward_impl(x, intermediate_x, insert_at)


def _convnext(
    arch: str,
    block_setting: List[CNBlockConfig],
    stochastic_depth_prob: float,
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ConvNeXt:
    model = ConvNeXt(block_setting, stochastic_depth_prob=stochastic_depth_prob, **kwargs)
    return model


def convnext_tiny(*, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    r"""ConvNeXt Tiny model architecture from the
    `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return _convnext("convnext_tiny", block_setting, stochastic_depth_prob, pretrained, progress, **kwargs)


def convnext_small(*, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    r"""ConvNeXt Small model architecture from the
    `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 27),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
    return _convnext("convnext_small", block_setting, stochastic_depth_prob, pretrained, progress, **kwargs)


def convnext_base(*, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    r"""ConvNeXt Base model architecture from the
    `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    block_setting = [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext("convnext_base", block_setting, stochastic_depth_prob, pretrained, progress, **kwargs)


def convnext_large(*, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    r"""ConvNeXt Large model architecture from the
    `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    block_setting = [
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 3),
        CNBlockConfig(768, 1536, 27),
        CNBlockConfig(1536, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext("convnext_large", block_setting, stochastic_depth_prob, pretrained, progress, **kwargs)
