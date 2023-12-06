# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) PyTorch Contributors 2022
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: BSD-3-Clause

# Code largely based on PyTorch https://github.com/pytorch/pytorch

import collections.abc
from contextlib import contextmanager
from itertools import repeat
from typing import Any, Dict, Optional, Tuple, TypeVar, Union

_SYM_NAME_STACK = []


def unique_syms_factory(tensor_cls):
    class unique_syms:
        def __init__(self):
            self.all_names = set()

        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            self.all_names = set()

        # noinspection PyPep8Naming
        def Tensor(self, *args, **kwargs):
            tensor = tensor_cls(*args, **kwargs)

            prev_len = len(self.all_names)
            self.all_names.update(tensor.names)
            if len(self.all_names) != prev_len + len(tensor.names):
                from collections import Counter

                print({k: v for k, v in Counter(tensor.names).items() if v != 1})
                raise ValueError(
                    f"Non-unique names!!! {len(self.all_names)} "
                    f"{prev_len} {len(tensor.names)} "
                    f"{len(set(tensor.names))}"
                )
            return tensor

    return unique_syms


@contextmanager
def sym_scope(name):
    try:
        _SYM_NAME_STACK.append(name)
        yield
    finally:
        _SYM_NAME_STACK.pop()


def get_sym_scope():
    return ".".join(_SYM_NAME_STACK)


V = TypeVar("V")
T = TypeVar("T")
_scalar_or_tuple_any_t = Union[T, Tuple[T, ...]]
_scalar_or_tuple_1_t = Union[T, Tuple[T]]
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_scalar_or_tuple_3_t = Union[T, Tuple[T, T, T]]
_scalar_or_tuple_4_t = Union[T, Tuple[T, T, T, T]]
_scalar_or_tuple_5_t = Union[T, Tuple[T, T, T, T, T]]
_scalar_or_tuple_6_t = Union[T, Tuple[T, T, T, T, T, T]]

# For arguments which represent size parameters (eg, kernel size, padding)
_size_any_t = _scalar_or_tuple_any_t[int]
_size_1_t = _scalar_or_tuple_1_t[int]
_size_2_t = _scalar_or_tuple_2_t[int]
_size_3_t = _scalar_or_tuple_3_t[int]
_size_4_t = _scalar_or_tuple_4_t[int]
_size_5_t = _scalar_or_tuple_5_t[int]
_size_6_t = _scalar_or_tuple_6_t[int]

# For arguments which represent optional size parameters (eg, adaptive pool
#  parameters)
_size_any_opt_t = _scalar_or_tuple_any_t[Optional[int]]
_size_2_opt_t = _scalar_or_tuple_2_t[Optional[int]]
_size_3_opt_t = _scalar_or_tuple_3_t[Optional[int]]

# For arguments that represent a ratio to adjust each dimension of an input
#  with (eg, upsampling parameters)
_ratio_2_t = _scalar_or_tuple_2_t[float]
_ratio_3_t = _scalar_or_tuple_3_t[float]
_ratio_any_t = _scalar_or_tuple_any_t[float]


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")
_quadruple = _ntuple(4, "_quadruple")


def _overwrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value " f"{new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
