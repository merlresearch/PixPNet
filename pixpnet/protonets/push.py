# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch

from pixpnet.data import SubsetWithIdx
from pixpnet.utils import get_logger
from pixpnet.utils_torch import unravel_index

logger = get_logger(__name__)


def push_prototypes(
    dataloader: SubsetWithIdx, protonet, class_specific=True, preprocess_func=None, duplicate_filter="sample"
):
    """push each prototype to the nearest patch in the training set"""
    was_training = protonet.training
    protonet.eval()

    prototype_shape = protonet.prototype_shape
    n_prototypes = protonet.num_prototypes
    prototype_layer_stride = protonet.prototype_layer_stride

    device = protonet.prototype_vectors.device
    dtype = protonet.prototype_vectors.dtype

    # saves the closest distance seen so far
    min_proto_dists = torch.full((n_prototypes,), torch.inf, dtype=dtype, device=device)
    # saves the patch representation that gives the current smallest distance
    min_fmap_patches = torch.zeros(prototype_shape, dtype=dtype, device=device)
    # saves the sample indices that each prototype corresponds to in dataloader
    min_sample_idxs = protonet.corresponding_sample_idxs
    # save the feature map indices
    min_fmap_idxs = protonet.min_fmap_idxs

    with torch.no_grad():
        # Find the closest training images to each prototype across the entire
        #  data set (updates closest each batch to achieve global maximums)
        for sample_idxs, x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            _update_prototypes_on_batch(
                sample_idxs=sample_idxs,
                x=x,
                y=y,
                protonet=protonet,
                min_proto_dists=min_proto_dists,
                min_fmap_patches=min_fmap_patches,
                min_sample_idxs=min_sample_idxs,
                min_fmap_idxs=min_fmap_idxs,
                class_specific=class_specific,
                preprocess_func=preprocess_func,
                proto_layer_stride=prototype_layer_stride,
                duplicate_filter=duplicate_filter,
            )

        q = torch.tensor([0, 0.25, 0.50, 0.75, 1], dtype=dtype, device=device)
        dist_percentiles = torch.quantile(min_proto_dists, q).tolist()
        logger.info(
            f"Prototypes pushing distances stats:\n"
            f'    {" / ".join(f"{x * 100:6.2f}%" for x in q.tolist())}\n'
            f'    {" / ".join(f"{x:7.4f}" for x in dist_percentiles)}\n'
            f"    {int(torch.isnan(min_proto_dists).sum())} / "
            f"{min_proto_dists.numel()} are NaN"
        )

        # Executing push...
        prototype_update = torch.reshape(min_fmap_patches, prototype_shape)

        proto_norm_pre = torch.norm(protonet.prototype_vectors)
        proto_norm_post = torch.norm(prototype_update)

        logger.info(
            f"Prototype vector Frobenius norm pre- and post-push: " f"{proto_norm_pre:.4f} --> {proto_norm_post:.4f}"
        )

        protonet.set_prototypes(
            prototype_update,  # P x D x K x K
            corresponding_sample_idxs=min_sample_idxs,  # P
            min_fmap_idxs=min_fmap_idxs,  # P x 4
        )

    if was_training:
        protonet.train()


def _update_prototypes_on_batch(
    sample_idxs,
    x,
    y,  # required if class_specific
    protonet,
    min_proto_dists,  # this will be updated
    min_fmap_patches,  # this will be updated
    min_sample_idxs,
    min_fmap_idxs,
    class_specific,
    preprocess_func,
    proto_layer_stride,
    duplicate_filter,
):
    """update each prototype for current search batch"""
    if preprocess_func is not None:
        x = preprocess_func(x)

    if duplicate_filter == "none":
        duplicate_filter = None

    features, proto_dists = protonet.push_forward(x)

    if class_specific:
        # Set dists of non-target prototypes to inf to prevent pushing
        #  non-class-specific prototypes
        prototype_classes = protonet.prototype_class_identity.argmax(dim=1)
        proto_dists[y[:, None] != prototype_classes[None]] = torch.inf

    n_prototypes, proto_dim, proto_h, proto_w = protonet.prototype_shape

    for j in range(n_prototypes):
        if duplicate_filter:
            # iterate by minimum distance to determine prototype. same prototype
            #  is avoided by setting all prototype distances to inf. we need to
            #  go by global minimum distance to determine which patch/sample
            #  should be assigned to which image, otherwise we can end up with
            #  non-min-dist pushing if iterating arbitrarily.
            min_proto_dist_idx = torch.argmin(proto_dists)
            min_proto_dist_idx = unravel_index(min_proto_dist_idx, proto_dists.size())
            min_proto_dist_j = proto_dists[min_proto_dist_idx].clone()
            # adjust j (the prototype idx) and grab sample/patch idxs
            idx, j, patch_idx_h, patch_idx_w = min_proto_dist_idx
            # mark all dists for prototype j off-limits from here-on-out (every
            #  other value is higher, so if the patch is not selected then
            #  nothing else possibly could be)
            proto_dists[:, j, :, :] = torch.inf
        else:
            # iterate prototype by prototype as we are not preventing duplicate
            #  prototypes
            proto_dist_j = proto_dists[:, j, :, :]
            min_proto_dist_j_idx = torch.argmin(proto_dist_j)
            idx, patch_idx_h, patch_idx_w = unravel_index(min_proto_dist_j_idx, proto_dist_j.size())
            min_proto_dist_j = proto_dist_j[idx, patch_idx_h, patch_idx_w]

        if min_proto_dist_j < min_proto_dists[j]:
            # retrieve the corresponding feature map patch
            fmap_h_start = patch_idx_h * proto_layer_stride
            fmap_h_end = fmap_h_start + proto_h
            fmap_w_start = patch_idx_w * proto_layer_stride
            fmap_w_end = fmap_w_start + proto_w
            # @formatter:off
            min_fmap_patch_j = features[idx, :, fmap_h_start:fmap_h_end, fmap_w_start:fmap_w_end]
            # @formatter:on

            # store the patch and its distance from prototype j
            min_proto_dists[j] = min_proto_dist_j
            min_fmap_patches[j] = min_fmap_patch_j
            min_sample_idxs[j] = sample_idxs[idx]
            min_fmap_idxs[j] = torch.tensor(
                [fmap_h_start, fmap_h_end, fmap_w_start, fmap_w_end],
                dtype=min_fmap_idxs.dtype,
                device=min_fmap_idxs.device,
            )

            # proto_dists: N x P x H x W
            if duplicate_filter:
                if duplicate_filter == "sample":
                    # mark all dists for sample off-limits
                    proto_dists[idx] = torch.inf
                elif duplicate_filter == "patch":
                    # mark all dists for sample patch off-limits
                    proto_dists[idx, :, patch_idx_h, patch_idx_w] = torch.inf
                else:
                    raise NotImplementedError(f"duplicate_filter = {duplicate_filter}")
