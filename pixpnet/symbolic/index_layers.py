# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) PyTorch Contributors 2022
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: BSD-3-Clause

# Some code based on PyTorch https://github.com/pytorch/pytorch

from itertools import chain, product
from math import ceil
from typing import List, Sequence, Tuple, Union

import numpy as np

# noinspection PyUnresolvedReferences
from pixpnet.symbolic.base_layers import (  # noqa: F401
    AdaptiveAvgPool2d_factory,
    AvgPool2d_factory,
    BatchNorm2d_factory,
    Conv2d_factory,
    Flatten,
    Linear_factory,
    MaxPool2d_factory,
    Module,
    ModuleDict,
    Sequential,
    _AdaptiveAvgPoolNd,
    _BatchNorm_factory,
    _ConvNd_factory,
    _MaxPoolNd,
    _NormBase_factory,
    cat,
    concat,
    end_index,
    flatten,
    np_sp_func,
    start_index,
    unsqueeze,
)
from pixpnet.symbolic.exceptions import NonPositiveDimError
from pixpnet.symbolic.misc import _pair, unique_syms_factory


class Tensor(np.ndarray):
    __slots__ = ("_names",)

    def __new__(cls, shape, name=None, **kwargs):
        obj = super().__new__(cls, shape=shape, dtype=object, **kwargs)

        names = [*product(*(range(d) for d in shape))]
        vcs = [
            HypercubeCollection.from_hypercube(Hypercube.from_slices(map(slice, indices)))
            for indices in product(*map(range, shape))
        ]
        obj.flat[:] = vcs
        obj._names = names
        return obj

    @property
    def names(self):
        return self._names


_FULL_SLICE = slice(None)


class OutputTensor(np.ndarray):
    __slots__ = "identical_channels", "_underlying"

    def __new__(cls, shape, dtype=None, identical_channels=False, **kwargs):
        if identical_channels:
            shape_orig = shape
            n, c, *dims = shape
            assert c > 0
            shape = (n, 1, *dims)
        obj = super().__new__(cls, shape=shape, dtype=dtype, **kwargs)
        obj._underlying = None
        if identical_channels:
            underlying = obj
            obj = np.broadcast_to(underlying, shape_orig, subok=True)
            obj._underlying = underlying
        obj.identical_channels = identical_channels
        return obj

    def __setitem__(self, key, value):
        if self._underlying is None:
            super().__setitem__(key, value)
        else:  # identical_channels memory view trick
            if len(key) >= 2:
                assert key[1] == _FULL_SLICE
            self._underlying[key] = value

    def __iadd__(self, other):
        if self._underlying is None:
            if self.flags["WRITEABLE"]:
                super().__iadd__(other)
            else:
                out = self + other
                if self.identical_channels and isinstance(other, OutputTensor) and other.identical_channels:
                    out.identical_channels = True
                return out
        else:  # identical_channels memory view trick
            if isinstance(other, OutputTensor) and other.identical_channels:
                self._underlying += other._underlying
            elif (isinstance(other, np.ndarray) and other.ndim >= 2 and other.shape[1] == 1) or not isinstance(
                other, np.ndarray
            ):
                self._underlying += other
            else:
                return self + other
        return self

    def __isub__(self, other):
        if self._underlying is None:
            if self.flags["WRITEABLE"]:
                super().__isub__(other)
            else:
                out = self - other
                if self.identical_channels and isinstance(other, OutputTensor) and other.identical_channels:
                    out.identical_channels = True
                return out
        else:  # identical_channels memory view trick
            if isinstance(other, OutputTensor) and other.identical_channels:
                self._underlying -= other._underlying
            elif (isinstance(other, np.ndarray) and other.ndim >= 2 and other.shape[1] == 1) or not isinstance(
                other, np.ndarray
            ):
                self._underlying -= other
            else:
                return self - other
        return self

    def __imul__(self, other):
        if self._underlying is None:
            if self.flags["WRITEABLE"]:
                super().__imul__(other)
            else:
                out = self * other
                if self.identical_channels and isinstance(other, OutputTensor) and other.identical_channels:
                    out.identical_channels = True
                return out
        else:  # identical_channels memory view trick
            if isinstance(other, OutputTensor) and other.identical_channels:
                self._underlying *= other._underlying
            elif (isinstance(other, np.ndarray) and other.ndim >= 2 and other.shape[1] == 1) or not isinstance(
                other, np.ndarray
            ):
                self._underlying *= other
            else:
                return self * other
        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return
        if not hasattr(self, "_underlying"):
            self._underlying = None
        if not hasattr(self, "identical_channels"):
            self.identical_channels = self.ndim >= 2 and self.shape[1] == 1


def has_identical_channels(tensor):
    return (isinstance(tensor, OutputTensor) and tensor.identical_channels) or (
        isinstance(tensor, np.ndarray) and tensor.ndim >= 2 and tensor.shape[1] == 1
    )


unique_syms = unique_syms_factory(Tensor)


class Hypercube:
    """NOTE: step is not supported in this function"""

    __slots__ = "indices", "ndim"

    indices: Tuple[Tuple[int, int], ...]
    ndim: int

    @classmethod
    def from_slices(
        cls,
        indices: Tuple[slice, ...],
    ) -> "Hypercube":
        indices = tuple((s.stop, s.stop + 1) if s.start is None else (s.start, s.stop) for s in indices)
        return cls(indices)

    def __init__(
        self,
        indices: Tuple[Tuple[int, int], ...],
    ):
        self.indices = indices
        self.ndim = len(self.indices)

    def serialize(self):
        return [[*idx] for idx in self.indices]

    @classmethod
    def deserialize(cls, indices):
        return cls(indices)

    def union(self, other: "Hypercube") -> Union["Hypercube", Tuple["Hypercube", "Hypercube"]]:
        if self == other:
            return self
        # contains_other  # self contains other
        # other_contains  # other contains self
        unequal_count = 0
        contains_other = other_contains = concat_dim = None
        for d in range(self.ndim):
            (r00, r01), (r10, r11) = self.indices[d], other.indices[d]
            r_int = r00 if r00 > r10 else r10, r01 if r01 < r11 else r11
            adjacent = (r00 == r11) or (r10 == r01)
            if not (len(r_int) or adjacent):  # no intersection, cannot combine
                return self, other
            unequal = (r00 != r10) or (r01 != r11)
            unequal_count += unequal
            if unequal:
                concat_dim = d
            if contains_other is None or contains_other:
                contains_other = r_int == (r10, r11)  # r1 contained within r0
            if other_contains is None or other_contains:
                other_contains = r_int == (r00, r01)  # r0 contained within r1
        if contains_other:
            return self
        if other_contains:
            return other
        if unequal_count == 1:
            # This means we can concatenate the hypercubes along a single axis
            (r00, r01), (r10, r11) = (self.indices[concat_dim], other.indices[concat_dim])
            indices = (
                self.indices[:concat_dim]
                + ((r00 if r00 < r10 else r10, r01 if r01 > r11 else r11),)
                + self.indices[concat_dim + 1 :]
            )
            return Hypercube(indices)

        return self, other

    def _intersection_indices(self: "Hypercube", other: "Hypercube"):
        indices = []
        for d in range(self.ndim):
            (r00, r01), (r10, r11) = self.indices[d], other.indices[d]
            r_int = r00 if r00 > r10 else r10, r01 if r01 < r11 else r11
            if not len(r_int):  # no intersection
                return None
            indices.append((r00 if r00 > r10 else r10, r01 if r01 < r11 else r11))

        return indices

    def intersection(self: "Hypercube", other: "Hypercube") -> Union["Hypercube", None]:
        indices = self._intersection_indices(other)
        return None if indices is None else Hypercube(indices)

    def atoms(self):
        return {*product(*(range(a, b) for a, b in self.indices))}

    def intersects(self, index: Tuple[int, ...]):
        return all(r0 <= index[d] < r1 for d, (r0, r1) in enumerate(self.indices))

    def corners(self, indices=None):
        if indices is None:
            indices = self.indices
        return product(*indices)

    def edges(self, indices=None):
        if indices is None:
            indices = self.indices
        flags = [(0, 1)] * (len(indices) - 1)
        # noinspection PyTypeChecker
        flags.append((0,))  # only one side so no duplicate edges
        for flags_i in product(*flags):
            corner = [idx[flag] for idx, flag in zip(indices, flags_i)]
            for j, flag in enumerate(flags_i):
                corner_other = corner.copy()
                corner_other[j] = indices[j][0 if flag else 1]
                yield corner, corner_other

    def difference(self, other):
        indices = self._intersection_indices(other)
        if indices is None:
            return self  # no intersection

        corners = self.corners()
        edges = self.edges()
        int_corners = self.corners(indices)
        int_edges = self.edges(indices)

        cubes = []

        # create cubes corner to corner (1:1 corners)
        for corner, int_corner in zip(corners, int_corners):
            indices_cube = []
            for d0, d1 in zip(corner, int_corner):
                if d0 > d1:
                    d0, d1 = d1, d0  # swap
                indices_cube.append((d0, d1))
            cubes.append(Hypercube(indices_cube))

        # create cubes edge to edge (1:1 edges)
        for edge, int_edge in zip(edges, int_edges):
            indices_cube = []
            for d0, d1 in zip(edge[0], int_edge[1]):
                if d0 > d1:
                    d0, d1 = d1, d0  # swap
                indices_cube.append((d0, d1))
            cubes.append(Hypercube(indices_cube))

        return HypercubeCollection.from_hypercubes(cubes)

    def as_slice(self, all_channels=False):  # -> Tuple[slice]
        return tuple(
            slice(None) if all_channels and i == 1 else slice(i_a, i_b) for i, (i_a, i_b) in enumerate(self.indices)
        )

    def take_from(self, arr, all_channels=False):
        assert self.ndim == arr.ndim
        return arr[self.as_slice(all_channels=all_channels)]

    def __len__(self):
        size = 1
        for r0, r1 in self.indices:
            size *= r1 - r0
        return size

    def __eq__(self, other):
        return self.indices == other.indices

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersection(other)

    def __sub__(self, other):
        return self.difference(other)

    def __repr__(self):
        return f"Hypercube(indices={self.indices})"

    def __str__(self):
        return repr(self)


class HypercubeCollection:
    __slots__ = ("hypercubes",)

    @classmethod
    def from_hypercube(cls, hypercube: Hypercube) -> "HypercubeCollection":
        obj = HypercubeCollection.__new__(cls)
        obj.hypercubes = [hypercube]
        return obj

    @classmethod
    def from_hypercubes(cls, hypercubes: Sequence[Hypercube]) -> "HypercubeCollection":
        obj = HypercubeCollection.__new__(cls)
        obj.hypercubes = hypercubes
        return obj

    def __init__(self, *args: Sequence["HypercubeCollection"]):
        hypercubes = [*chain.from_iterable(hc.hypercubes for hc in chain.from_iterable(args))]
        self.hypercubes = self._reduce_hcs(hypercubes)

    def serialize(self):
        return [hc.serialize() for hc in self.hypercubes]

    @classmethod
    def deserialize(cls, arr_indices):
        return cls.from_hypercubes([Hypercube.deserialize(indices) for indices in arr_indices])

    @staticmethod
    def _reduce_hcs(hypercubes: List[Hypercube]) -> List[Hypercube]:
        uncombined = []
        combined = []
        while hypercubes:
            hc0 = hypercubes.pop()
            HypercubeCollection._compare_hcs(
                hc0, compare_src=hypercubes, combined_dest=combined, uncombined_dest=uncombined
            )
        while combined:
            hc0 = combined.pop()
            HypercubeCollection._compare_hcs(
                hc0, compare_src=uncombined, combined_dest=combined, uncombined_dest=uncombined
            )
        return uncombined

    @staticmethod
    def _compare_hcs(hc0: Hypercube, compare_src: List, combined_dest: List, uncombined_dest: List):
        idxs_to_drop = []
        for i, hc1 in enumerate(compare_src):
            hcc = hc0 | hc1
            if not isinstance(hcc, tuple):
                combined_dest.append(hcc)
                idxs_to_drop.append(i)
                break
        else:
            uncombined_dest.append(hc0)
        for i in idxs_to_drop:
            del compare_src[i]

    def atoms(self, *args):
        if args:
            print("ignored:", args)
        return set().union(*(hc.atoms() for hc in self.hypercubes))

    def intersects(self, index: Tuple[int]):
        for hc in self.hypercubes:
            if hc.intersects(index):
                return True
        return False

    def intersecting_indices(self, indices: Sequence[Tuple[int]]):
        return [index for index in indices if self.intersects(index)]

    def difference(self, other):
        if isinstance(other, Hypercube):
            hypercubes = [hc.difference(other) for hc in self.hypercubes]
        else:
            hypercubes = [hc.difference(hc_other) for hc in self.hypercubes for hc_other in other.hypercubes]
        return HypercubeCollection(hypercubes)

    def take_from(self, arr, all_channels=False):
        to_intersect = []
        for hc in self.hypercubes:
            for hc_i in to_intersect:
                hc = hc.difference(hc_i)
            to_intersect.append(hc)
        if len(to_intersect) == 1:
            return to_intersect[0].take_from(arr, all_channels=all_channels)
        return [to_int.take_from(arr, all_channels=all_channels) for to_int in to_intersect]

    def as_slices(self, all_channels=False):
        to_intersect = []
        for hc in self.hypercubes:
            for hc_i in to_intersect:
                hc = hc.difference(hc_i)
            to_intersect.append(hc)
        return [hc.as_slice(all_channels=all_channels) for hc in to_intersect]

    def __len__(self):
        size = 0
        to_intersect = []
        for hc in self.hypercubes:
            for hc_i in to_intersect:
                hc = hc.difference(hc_i)
            size += len(hc)
            to_intersect.append(hc)
        return size

    def __or__(self, other: "HypercubeCollection") -> "HypercubeCollection":
        """Union operator ``self | other``"""
        return HypercubeCollection((self, other))

    def __add__(self, other: "HypercubeCollection") -> "HypercubeCollection":
        return HypercubeCollection((self, other))

    def __mul__(self, other: "HypercubeCollection") -> "HypercubeCollection":
        return HypercubeCollection((self, other))

    def __repr__(self):
        return f"HypercubeCollection(hypercubes={self.hypercubes})"

    def __str__(self):
        return repr(self)


class ReLU(Module):
    __slots__ = ()

    def __init__(self, inplace=False):
        super(ReLU, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input


SiLU = ReLU
Sigmoid = ReLU
GELU = ReLU


def conv2d(input, weight, bias, stride, padding=0, dilation=1, groups=1):
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    if not (dh == dw == 1):
        raise NotImplementedError

    c_out, ci_per_group, kh, kw = weight.shape
    co_per_group = c_out // groups
    h_in, w_in = input.shape[-2:]
    h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    single_sample = input.ndim == 3
    if single_sample:
        input = input.reshape((1, *input.shape))
    if any(d <= 0 for d in (c_out, h_out, w_out)):
        raise NonPositiveDimError((c_out, h_out, w_out))
    # input has identical channels or no groups or there is only 1 channel
    identical_channels = has_identical_channels(input)
    output = OutputTensor(
        (input.shape[0], c_out, h_out, w_out), dtype=input.dtype, identical_channels=identical_channels or groups == 1
    )
    if not output.identical_channels:
        print(
            "WARNING: input does not have identical channels in conv2d and "
            "groups != 1, this will take longer to compute"
        )

    for oh in range(h_out):
        ih0 = (oh * sh) - ph
        ih1 = ih0 + kh
        if ih0 < 0:
            ih0 = 0
        for ow in range(w_out):
            iw0 = (ow * sw) - pw
            iw1 = iw0 + kw
            if iw0 < 0:
                iw0 = 0
            # slice:  n x c x kh x kw
            # weight: c_out x c x kh x kw
            if identical_channels:
                # we can ignore groups. take first channel (arbitrary as all
                #  channels are the same)
                x_slice = input[:, 0, ih0:ih1, iw0:iw1]
                for n in range(output.shape[0]):
                    hc_n = HypercubeCollection(x_slice[n].flatten().tolist())
                    output[n, :, oh, ow] = hc_n
            elif groups == 1:
                x_slice = input[:, :, ih0:ih1, iw0:iw1]
                for n in range(output.shape[0]):
                    hc_n = HypercubeCollection(x_slice[n].flatten().tolist())
                    output[n, :, oh, ow] = hc_n
            else:
                for g in range(groups):
                    co0 = g * co_per_group
                    co1 = co0 + co_per_group
                    ci0 = g * ci_per_group
                    ci1 = ci0 + ci_per_group
                    x_slice = input[:, ci0:ci1, ih0:ih1, iw0:iw1]
                    for n in range(output.shape[0]):
                        hc_n = HypercubeCollection(x_slice[n, :].flatten().tolist())
                        for c in range(co0, co1):
                            output[n, c, oh, ow] = hc_n

    if single_sample:
        output = output.squeeze(axis=0)
    return output


def linear(input: Tensor, weight: Tensor, bias: Tensor) -> np.ndarray:
    # out_features x in_features
    output = OutputTensor((input.shape[0], weight.shape[0]), dtype=input.dtype, identical_channels=True)

    for n in range(output.shape[0]):
        hc_n = HypercubeCollection(input[n].tolist())
        output[n, :] = hc_n

    return output


def adaptive_avg_pool_2d(
    input: np.ndarray,  # Tensor
    output_size=None,
):
    input_height, input_width = input.shape[-2:]
    output_height, output_width = _pair(output_size)
    output_height = output_height or input_height
    output_width = output_width or input_width

    single_sample = input.ndim == 3
    if single_sample:
        input = input.reshape((1, *input.shape))

    identical_channels = has_identical_channels(input)
    output = OutputTensor(
        (input.shape[0], input.shape[1], output_height, output_width),
        dtype=input.dtype,
        identical_channels=identical_channels,
    )
    if not identical_channels:
        print(
            "WARNING: input does not have identical channels in "
            "adaptive_avg_pool_2d, this will take longer to compute"
        )

    for oh in range(output_height):
        ih0 = start_index(oh, output_height, input_height)
        ih1 = end_index(oh, output_height, input_height)

        for ow in range(output_width):
            iw0 = start_index(ow, output_width, input_width)
            iw1 = end_index(ow, output_width, input_width)

            x_slice = input[:, :, ih0:ih1, iw0:iw1]
            if identical_channels:
                for n in range(input.shape[0]):
                    # arbitrarily take first channel
                    output[n, :, oh, ow] = HypercubeCollection(x_slice[n, 0].flatten().tolist())
            else:
                for c in range(input.shape[1]):
                    for n in range(input.shape[0]):
                        output[n, c, oh, ow] = HypercubeCollection(x_slice[n, c].flatten().tolist())

    if single_sample:
        output = output.squeeze(axis=0)
    return output


def max_pool2d(
    input, kernel_size, stride, padding=0, dilation=1, return_indices: bool = False, ceil_mode: bool = False
):
    if return_indices:
        raise NotImplementedError
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    if not (dh == dw == 1):
        raise NotImplementedError
    h_in, w_in = input.shape[-2:]
    h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) / sh + 1
    w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) / sw + 1
    h_out = ceil(h_out) if ceil_mode else int(h_out)
    w_out = ceil(w_out) if ceil_mode else int(w_out)

    single_sample = input.ndim == 3
    if single_sample:
        input = input.reshape((1, *input.shape))

    identical_channels = has_identical_channels(input)
    output = OutputTensor(
        (input.shape[0], input.shape[1], h_out, w_out), dtype=input.dtype, identical_channels=identical_channels
    )
    if not identical_channels:
        print("WARNING: input does not have identical channels in " "max_pool2d, this will take longer to compute")
    for oh in range(h_out):
        ih0 = (oh * sh) - ph
        ih1 = ih0 + kh
        ih0 = max(ih0, 0)
        for ow in range(w_out):
            iw0 = (ow * sw) - pw
            iw1 = iw0 + kw
            iw0 = max(iw0, 0)
            if identical_channels:
                for n in range(output.shape[0]):
                    # arbitrarily take first channel
                    values = input[n, 0, ih0:ih1, iw0:iw1].flatten().tolist()
                    output[n, :, oh, ow] = HypercubeCollection(values)
            else:
                for n in range(output.shape[0]):
                    for c in range(output.shape[1]):
                        values = input[n, c, ih0:ih1, iw0:iw1].flatten().tolist()
                        output[n, c, oh, ow] = HypercubeCollection(values)

    if single_sample:
        output = output.squeeze(axis=0)
    return output


def avg_pool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    h_in, w_in = input.shape[-2:]
    h_out = (h_in + 2 * ph - kh) / sh + 1
    w_out = (w_in + 2 * pw - kw) / sw + 1
    h_out = ceil(h_out) if ceil_mode else int(h_out)
    w_out = ceil(w_out) if ceil_mode else int(w_out)

    single_sample = input.ndim == 3
    if single_sample:
        input = input.reshape((1, *input.shape))

    identical_channels = has_identical_channels(input)
    output = OutputTensor(
        (input.shape[0], input.shape[1], h_out, w_out), dtype=input.dtype, identical_channels=identical_channels
    )
    if not identical_channels:
        print("WARNING: input does not have identical channels in " "avg_pool2d, this will take longer to compute")
    for oh in range(h_out):
        ih0 = (oh * sh) - ph
        ih1 = ih0 + kh
        ih0 = max(ih0, 0)
        for ow in range(w_out):
            iw0 = (ow * sw) - pw
            iw1 = iw0 + kw
            iw0 = max(iw0, 0)
            if identical_channels:
                for n in range(output.shape[0]):
                    # arbitrarily take first channel
                    values = input[n, 0, ih0:ih1, iw0:iw1].flatten().tolist()
                    output[n, :, oh, ow] = HypercubeCollection(values)
            else:
                for n in range(output.shape[0]):
                    for c in range(output.shape[1]):
                        values = input[n, c, ih0:ih1, iw0:iw1].flatten().tolist()
                        output[n, c, oh, ow] = HypercubeCollection(values)

    if single_sample:
        output = output.squeeze(axis=0)
    return output


def batch_norm(input, running_mean, running_var, momentum, eps, affine):
    return input


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(LayerNorm, self).__init__()
        import numbers

        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine

    def forward(self, input: Tensor) -> Tensor:
        return input


_ConvNd = _ConvNd_factory(Tensor, parameters=False)
Conv2d = Conv2d_factory(_ConvNd, conv2d)
_NormBase = _NormBase_factory(Tensor)
_BatchNorm = _BatchNorm_factory(_NormBase, batch_norm)
BatchNorm2d = BatchNorm2d_factory(_BatchNorm)

Linear = Linear_factory(Tensor, linear, parameters=False)
AdaptiveAvgPool2d = AdaptiveAvgPool2d_factory(adaptive_avg_pool_2d)
MaxPool2d = MaxPool2d_factory(max_pool2d)
AvgPool2d = AvgPool2d_factory(avg_pool2d)
