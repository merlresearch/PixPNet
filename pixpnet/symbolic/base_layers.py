# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) PyTorch Contributors 2022
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: BSD-3-Clause

# Code largely based on PyTorch https://github.com/pytorch/pytorch

from abc import ABCMeta
from collections import OrderedDict
from functools import wraps
from math import ceil
from typing import Any, Optional, Tuple, Union

import numpy as np

from pixpnet.symbolic.misc import _pair, _reverse_repeat_tuple, _size_2_t, sym_scope


def flatten(input, start_dim: int = 0, end_dim: int = -1):
    ndim = len(input.shape)
    if start_dim < 0:
        start_dim += ndim + 1
        assert start_dim >= 0
    if end_dim < 0:
        end_dim += ndim + 1
        assert end_dim >= 0
    assert start_dim <= end_dim
    if start_dim == end_dim:
        return input
    flat_size = 1
    for d in input.shape[start_dim : end_dim + 1]:
        flat_size *= d
    out_shape = input.shape[0:start_dim] + (flat_size,) + input.shape[end_dim + 1 :]
    return input.reshape(*out_shape)


def cat(tensors, dim=0):
    return np.concatenate(tensors, axis=dim)


concat = cat


def unsqueeze(x, dim):
    return np.expand_dims(x, axis=dim)


def np_sp_func(func):
    @wraps(func)
    def wrapper(*input, **kwargs):
        out = func(*input, **kwargs)
        return out

    return wrapper


class Module:
    def __init__(self) -> None:
        pass

    def forward(self, *input: Any) -> Any:
        raise NotImplementedError

    def __call__(self, *input, **kwargs):
        # noinspection PyArgumentList

        return self.forward(*input, **kwargs)

    def add_module(self, name, module):
        if hasattr(self, name):
            raise ValueError(f"{self} already has attribute {name}")
        setattr(self, name, module)


class ModuleDict(Module):
    def __init__(self, module_dict=None):
        super(ModuleDict, self).__init__()
        self.__module_dict = module_dict or OrderedDict()

    def values(self):
        return self.__module_dict.values()

    def items(self):
        return self.__module_dict.items()

    def add_module(self, name, module):
        super(ModuleDict, self).add_module(name, module)
        self.__module_dict[name] = module


class Sequential(ModuleDict):
    def __init__(self, *modules):
        if len(modules) == 1 and isinstance(modules[0], OrderedDict):
            modules = modules[0]
        else:
            modules = OrderedDict((str(i), module) for i, module in enumerate(modules))
        super().__init__(modules)

    def __iter__(self):
        yield from self.values()

    def forward(self, input):
        for module in self:
            input = module(input)
        return input


class DummyTensor:
    __slots__ = ("shape",)

    def __init__(self, shape):
        self.shape = shape


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return flatten(input, start_dim=self.start_dim, end_dim=self.end_dim)


def _ConvNd_factory(tensor_cls, parameters=True):
    tensor_cls = tensor_cls if parameters else DummyTensor

    class _ConvNd(Module, metaclass=ABCMeta):
        def _conv_forward(self, input, weight, bias):
            raise NotImplementedError

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, ...],
            stride: Tuple[int, ...],
            padding: Tuple[int, ...],
            dilation: Tuple[int, ...],
            transposed: bool,
            output_padding: Tuple[int, ...],
            groups: int,
            bias: bool,
            padding_mode: str,
        ) -> None:
            super(_ConvNd, self).__init__()
            if in_channels % groups != 0:
                raise ValueError("in_channels must be divisible by groups")
            if out_channels % groups != 0:
                raise ValueError("out_channels must be divisible by groups")
            if in_channels == 0:
                raise ValueError(f"in_channels={in_channels}")
            if out_channels == 0:
                raise ValueError(f"out_channels={out_channels}")
            valid_padding_strings = {"same", "valid"}
            if isinstance(padding, str):
                if padding not in valid_padding_strings:
                    raise ValueError(
                        "Invalid padding string {!r}, should be one of " "{}".format(padding, valid_padding_strings)
                    )
                if padding == "same" and any(s != 1 for s in stride):
                    raise ValueError("padding='same' is not supported for " "strided convolutions")

            valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
            if padding_mode not in valid_padding_modes:
                raise ValueError(
                    "padding_mode must be one of {}, but got "
                    "padding_mode='{}'".format(valid_padding_modes, padding_mode)
                )
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.transposed = transposed
            self.output_padding = output_padding
            self.groups = groups
            self.padding_mode = padding_mode
            # `_reversed_padding_repeated_twice` is the padding to be passed to
            # `F.pad` if needed (e.g., for non-zero padding types that are
            # implemented as two ops: padding + conv). `F.pad` accepts paddings
            # in reverse order than the dimension.
            if isinstance(self.padding, str):
                self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
                if padding == "same":
                    for d, k, i in zip(dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)):
                        total_padding = d * (k - 1)
                        left_pad = total_padding // 2
                        self._reversed_padding_repeated_twice[2 * i] = left_pad
                        self._reversed_padding_repeated_twice[2 * i + 1] = total_padding - left_pad
            else:
                self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

            if transposed:
                with sym_scope("weight"):
                    self.weight = tensor_cls(shape=(in_channels, out_channels // groups, *kernel_size))
            else:
                with sym_scope("weight"):
                    self.weight = tensor_cls(shape=(out_channels, in_channels // groups, *kernel_size))
            if bias:
                with sym_scope("bias"):
                    self.bias = tensor_cls(shape=(out_channels,))
            else:
                self.bias = None

    return _ConvNd


def Conv2d_factory(_ConvNd, conv2d):
    class Conv2d(_ConvNd):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",
        ) -> None:
            kernel_size_ = _pair(kernel_size)
            stride_ = _pair(stride)
            padding_ = padding if isinstance(padding, str) else _pair(padding)
            dilation_ = _pair(dilation)
            super(Conv2d, self).__init__(
                in_channels,
                out_channels,
                kernel_size_,
                stride_,
                padding_,
                dilation_,
                False,
                _pair(0),
                groups,
                bias,
                padding_mode,
            )

        def _conv_forward(self, input, weight, bias):
            assert input.shape[1] // self.groups == weight.shape[1], (input.shape[1], self.groups, weight.shape[1])
            if self.padding_mode != "zeros":
                raise NotImplementedError(self.padding_mode)
            return conv2d(
                input,
                weight,
                bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        def forward(self, input):
            return self._conv_forward(input, self.weight, self.bias)

    return Conv2d


def _NormBase_factory(tensor_cls, parameters=True):
    tensor_cls = tensor_cls if parameters else DummyTensor

    class _NormBase(Module, metaclass=ABCMeta):
        """Common base of _InstanceNorm and _BatchNorm"""

        def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
        ) -> None:
            super(_NormBase, self).__init__()
            self.num_features = num_features
            self.eps = eps
            self.momentum = momentum
            self.affine = affine
            self.track_running_stats = track_running_stats
            if self.track_running_stats:
                with sym_scope("running_mean"):
                    self.running_mean = tensor_cls(shape=(num_features,))
                with sym_scope("running_var"):
                    self.running_var = tensor_cls(shape=(num_features,))
            else:
                self.running_mean = None
                self.running_var = None
            if self.affine:
                with sym_scope("weight"):
                    self.weight = tensor_cls(shape=(num_features,))
                with sym_scope("bias"):
                    self.bias = tensor_cls(shape=(num_features,))
            else:
                self.weight = None
                self.bias = None

        def _check_input_dim(self, input):
            raise NotImplementedError

    return _NormBase


def _BatchNorm_factory(_NormBase, batch_norm=None):
    class _BatchNorm(_NormBase, metaclass=ABCMeta):
        def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        ):
            super(_BatchNorm, self).__init__(
                num_features,
                eps,
                momentum,
                affine,
                track_running_stats,
            )

        def forward(self, input):
            self._check_input_dim(input)
            if batch_norm is None:
                raise NotImplementedError
            return batch_norm(input, self.running_mean, self.running_var, self.momentum, self.eps, self.affine)

    return _BatchNorm


def BatchNorm2d_factory(_BatchNorm):
    class BatchNorm2d(_BatchNorm):
        @staticmethod
        def _check_input_dim(input):
            ndim = len(input.shape)
            if ndim != 4:
                raise ValueError("expected 4D input (got {}D input)".format(ndim))

    return BatchNorm2d


def Linear_factory(tensor_cls, linear, parameters=True):
    tensor_cls = tensor_cls if parameters else DummyTensor

    class Linear(Module):
        def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
            super(Linear, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            with sym_scope("weight"):
                self.weight = tensor_cls(shape=(out_features, in_features))
            if bias:
                with sym_scope("bias"):
                    self.bias = tensor_cls(shape=(out_features,))
            else:
                self.bias = None

        def forward(self, input):
            assert input.shape[-1] == self.weight.shape[1]
            orig_shape = None
            if input.ndim != 2:
                orig_shape = input.shape
                input = input.reshape(-1, input.shape[-1])
            out = linear(input, self.weight, self.bias)
            if orig_shape is not None:
                out = out.reshape(*orig_shape[:-1], out.shape[-1])
            return out

    return Linear


class _AdaptiveAvgPoolNd(Module, metaclass=ABCMeta):
    def __init__(self, output_size) -> None:
        super(_AdaptiveAvgPoolNd, self).__init__()
        self.output_size = output_size

    def extra_repr(self) -> str:
        return "output_size={}".format(self.output_size)


def AdaptiveAvgPool2d_factory(adaptive_avg_pool_2d):
    class AdaptiveAvgPool2d(_AdaptiveAvgPoolNd):
        def forward(self, input):
            return adaptive_avg_pool_2d(input, output_size=self.output_size)

    return AdaptiveAvgPool2d


def start_index(out_idx: int, out_len: int, in_len: int) -> int:
    """
    * out_idx: the current index of output matrix
    * out_len: the dimension_size of output matrix
    * in_len: the dimension_size of input matrix
    * Basically, in_len / out_len gives the number of
    * elements in each average computation.
    * This function computes the start index on input matrix.
    """
    return (out_idx * in_len) // out_len  # floor


def end_index(out_idx: int, out_len: int, in_len: int) -> int:
    """
    * Parameter definition is the same as start_index.
    * This function computes the end index on input matrix.
    """
    return ceil(((out_idx + 1) * in_len) / out_len)


class _MaxPoolNd(Module, metaclass=ABCMeta):
    def __init__(
        self, kernel_size, stride=None, padding=0, dilation=1, return_indices: bool = False, ceil_mode: bool = False
    ) -> None:
        super(_MaxPoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode


def MaxPool2d_factory(max_pool2d):
    class MaxPool2d(_MaxPoolNd):
        def forward(self, input):
            return max_pool2d(
                input,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                ceil_mode=self.ceil_mode,
                return_indices=self.return_indices,
            )

    return MaxPool2d


class _AvgPoolNd(Module, metaclass=ABCMeta):
    pass


def AvgPool2d_factory(avg_pool2d):
    class AvgPool2d(_AvgPoolNd):
        def __init__(
            self,
            kernel_size: _size_2_t,
            stride: Optional[_size_2_t] = None,
            padding: _size_2_t = 0,
            ceil_mode: bool = False,
            count_include_pad: bool = True,
            divisor_override: Optional[int] = None,
        ) -> None:
            super(AvgPool2d, self).__init__()
            self.kernel_size = kernel_size
            self.stride = stride if (stride is not None) else kernel_size
            self.padding = padding
            self.ceil_mode = ceil_mode
            self.count_include_pad = count_include_pad
            self.divisor_override = divisor_override

        def forward(self, input):
            return avg_pool2d(
                input,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                self.count_include_pad,
                self.divisor_override,
            )

    return AvgPool2d
