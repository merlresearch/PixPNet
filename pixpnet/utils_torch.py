# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) PyTorch Contributors 2022
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: BSD-3-Clause

import operator
from functools import reduce
from typing import Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.testing._internal.common_dtype import integral_types

from pixpnet.utils import get_logger

logger = get_logger(__name__)


# PyTorch native unravel_index issue is stale
# https://github.com/pytorch/pytorch/issues/35674
# Borrowed from stale PR:
# https://github.com/pytorch/pytorch/pull/66687
def unravel_index(
    indices: Tensor, shape: Union[int, Sequence, Tensor], *, as_tuple: bool = True
) -> Union[Tuple[Tensor, ...], Tensor]:
    r"""Converts a `Tensor` of flat indices into a `Tensor` of coordinates for
    the given target shape.
    Args:
        indices: An integral `Tensor` containing flattened indices of a `Tensor`
                 of dimension `shape`.
        shape: The shape (can be an `int`, a `Sequence` or a `Tensor`) of the
               `Tensor` for which
               the flattened `indices` are unraveled.
        as_tuple: A boolean value, which if `True` will return the result as
                  tuple of Tensors,
                  else a `Tensor` will be returned. Default: `True`
    Returns:
        unraveled coordinates from the given `indices` and `shape`. See
        description of `as_tuple` for returning a `tuple`.
    .. note:: The default behaviour of this function is analogous to
              `numpy.unravel_index
              <https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html>`_.
    Example::
        >>> indices = torch.tensor([22, 41, 37])
        >>> shape = (7, 6)
        >>> unravel_index(indices, shape)
        (tensor([3, 6, 6]), tensor([4, 5, 1]))
        >>> unravel_index(indices, shape, as_tuple=False)
        tensor([[3, 4],
                [6, 5],
                [6, 1]])
        >>> indices = torch.tensor([3, 10, 12])
        >>> shape_ = (4, 2, 3)
        >>> unravel_index(indices, shape_)
        (tensor([0, 1, 2]), tensor([1, 1, 0]), tensor([0, 1, 0]))
        >>> unravel_index(indices, shape_, as_tuple=False)
        tensor([[0, 1, 0],
                [1, 1, 1],
                [2, 0, 0]])
    """

    def _helper_type_check(inp: Union[int, Sequence, Tensor], name: str):
        # `indices` is expected to be a tensor, while `shape` can be a
        #  sequence/int/tensor
        if name == "shape" and isinstance(inp, Sequence):
            for dim in inp:
                if not isinstance(dim, int):
                    raise TypeError("Expected shape to have only integral elements.")
                if dim < 0:
                    raise ValueError("Negative values in shape are not allowed.")
        elif name == "shape" and isinstance(inp, int):
            if inp < 0:
                raise ValueError("Negative values in shape are not allowed.")
        elif isinstance(inp, Tensor):
            if inp.dtype not in integral_types():
                raise TypeError(f"Expected {name} to be an integral tensor, " f"but found dtype: {inp.dtype}")
            if torch.any(inp < 0):
                raise ValueError(f"Negative values in {name} are not allowed.")
        else:
            allowed_types = "Sequence/Scalar (int)/Tensor" if name == "shape" else "Tensor"
            msg = f"{name} should either be a {allowed_types}, but found " f"{type(inp)}"
            raise TypeError(msg)

    _helper_type_check(indices, "indices")
    _helper_type_check(shape, "shape")

    # Convert to a tensor, with the same properties as that of indices
    if isinstance(shape, Sequence):
        shape_tensor: Tensor = indices.new_tensor(shape)
    elif isinstance(shape, int) or (isinstance(shape, Tensor) and shape.ndim == 0):
        shape_tensor = indices.new_tensor((shape,))
    else:
        shape_tensor = shape

    # By this time, shape tensor will have dim = 1 if it was passed as scalar
    #  (see if-elif above)
    assert shape_tensor.ndim == 1, (
        f"Expected dimension of shape tensor to be <= 1, but got the tensor " f"with dim: {shape_tensor.ndim}."
    )

    if indices.numel() == 0:
        shape_numel = shape_tensor.numel()
        if shape_numel == 0:
            raise ValueError("Got indices and shape as empty tensors, expected " "non-empty tensors.")
        else:
            output = [indices.new_tensor([]) for _ in range(shape_numel)]
            return tuple(output) if as_tuple else torch.stack(output, dim=1)

    if torch.max(indices) >= torch.prod(shape_tensor):
        raise ValueError("Target shape should cover all source indices.")

    coefs = shape_tensor[1:].flipud().cumprod(dim=0).flipud()
    coefs = torch.cat((coefs, coefs.new_tensor((1,))), dim=0)
    coords = torch.div(indices[..., None], coefs, rounding_mode="trunc") % shape_tensor

    if as_tuple:
        return tuple(coords[..., i] for i in range(coords.size(-1)))
    return coords


def take_rf_from_bbox(rf_layer, fmap_h_start, fmap_w_start, proto_h, proto_w, proto_layer_stride):
    # retrieve the corresponding feature map patch
    fmap_h_start *= proto_layer_stride
    fmap_h_end = fmap_h_start + proto_h
    fmap_w_start *= proto_layer_stride
    fmap_w_end = fmap_w_start + proto_w
    return take_rf(rf_layer, fmap_h_start, fmap_h_end, fmap_w_start, fmap_w_end)


def take_rf(rf_layer, fmap_h_start, fmap_h_end, fmap_w_start, fmap_w_end):
    # @formatter:off
    rf_feat = rf_layer[0, :, fmap_h_start:fmap_h_end, fmap_w_start:fmap_w_end]
    # @formatter:on
    identical_channels = getattr(rf_feat, "identical_channels", None)
    if identical_channels is None:
        logger.warning("BIST: rf_layer did not have the identical_channels " "attribute!!!")
    if identical_channels:
        rf_feat = rf_feat[0]
    elif rf_feat.shape[0] != 1:
        logger.info("rf_feat does not have identical channels so we are " "taking the union of all channel elements!!!")
        rf_feat = rf_feat.copy()
        orig_shape = rf_feat.shape
        rf_feat = rf_feat.reshape(orig_shape[0], -1)
        for i in range(rf_feat.shape[1]):
            rf_feat[:, i] = reduce(operator.or_, rf_feat[:, i])
        rf_feat = rf_feat.reshape(orig_shape)
        # take the first channel as all channels are the same now
        rf_feat = rf_feat[0, ...]
    if rf_feat.size == 1:
        rf_feat = rf_feat.item()
    else:
        raise NotImplementedError(
            "rf_feat has a size > 1, this case is not handled right now. "
            f"rf_layer.shape={rf_layer.shape} rf_feat.shape={rf_feat.shape} "
            f"rf_feat.identical_channels={identical_channels}"
            f"fmap_h_start,fmap_h_end,fmap_w_start,fmap_w_end="
            f"{(fmap_h_start, fmap_h_end, fmap_w_start, fmap_w_end)}\n"
            f"rf_feat: {rf_feat}"
        )
    return rf_feat


def slices_to_bboxes(slices: Sequence[Tuple[slice, slice, slice, slice]]):
    bboxes = []
    for slices_i in slices:
        n_s, c_s, h_s, w_s = slices_i
        xy = (w_s.start, h_s.start)
        width = w_s.stop - w_s.start
        height = h_s.stop - h_s.start
        bboxes.append((xy, width, height))
    return bboxes
