# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.ops.boxes import box_iou
from tqdm.auto import tqdm

from pixpnet import data as pp_data
from pixpnet.protonets.models.protonet import ProtoNet
from pixpnet.protonets.prp.prp import generate_prp_image
from pixpnet.utils import get_logger
from pixpnet.utils_torch import unravel_index

logger = get_logger(__name__)


def _resize_xy(xs, ys, im_h, im_w, resize_h, resize_w):
    # linearly resize xs and ys
    ratio_h = resize_h / im_h
    ratio_w = resize_w / im_w
    return xs * ratio_w, ys * ratio_h


def _preprocess_part_annotations_img(
    orig_shape,
    parts,
    metadata,
    img_id,
    crop_size,
    resize,
    downsample,
    bbox_cropped,
):
    # preprocess part xy information
    parts_img = parts[parts["img_id"] == img_id.item()]
    if parts_img.empty:
        logger.info(f"parts_img is empty for img_id={img_id}!!! " f"Continuing...")
        return None, None

    parts_x = torch.tensor(parts_img.x.values)
    parts_y = torch.tensor(parts_img.y.values)
    part_ids_img = parts_img["part_id"].values

    _, im_h, im_w = orig_shape
    im_h, im_w = im_h.item(), im_w.item()

    if bbox_cropped:
        metadata_i = metadata[metadata["img_id"] == img_id.item()]
        assert len(metadata_i) == 1
        metadata_i = metadata_i.iloc[0]
        parts_x -= metadata_i.x
        parts_y -= metadata_i.y
        im_h, im_w = metadata_i.h, metadata_i.w

    crop_h = crop_w = crop_size
    if crop_size:
        if min(im_h, im_w) < crop_size:
            ratio = max(crop_size / im_h, crop_size / im_w)
            crop_resize_h = round(im_h * ratio)
            crop_resize_w = round(im_w * ratio)
            # resize
            parts_x, parts_y = _resize_xy(
                xs=parts_x, ys=parts_y, im_h=im_h, im_w=im_w, resize_h=crop_resize_h, resize_w=crop_resize_w
            )
            # update im_h and im_w
            im_h, im_w = crop_resize_h, crop_resize_w
    if resize:
        # resize
        parts_x, parts_y = _resize_xy(xs=parts_x, ys=parts_y, im_h=im_h, im_w=im_w, resize_h=resize, resize_w=resize)
        # update im_h and im_w
        im_h, im_w = resize, resize
    if crop_size:
        if crop_w > im_w or crop_h > im_h:
            raise NotImplementedError

        crop_top = int(round((im_h - crop_h) / 2.0))
        crop_left = int(round((im_w - crop_w) / 2.0))
        # crop
        parts_y -= crop_top
        parts_x -= crop_left
        # update im_h and im_w
        im_h, im_w = crop_h, crop_w

    if downsample:
        # resize
        parts_x, parts_y = _resize_xy(
            xs=parts_x, ys=parts_y, im_h=im_h, im_w=im_w, resize_h=downsample, resize_w=downsample
        )
        # update im_h and im_w
        im_h, im_w = downsample, downsample

    # validate
    invisible = ((parts_y < 0) | (parts_y >= im_h)) | ((parts_x < 0) | (parts_x >= im_w))
    if invisible.any():
        logger.warning(
            f"Due to preprocessing of (img_id={img_id},"
            f"part_ids={part_ids_img[invisible]}), "
            f"{invisible.sum()}/{len(invisible)} object "
            f"parts became invisible"
        )
        visible = ~invisible
        parts_x, parts_y = parts_x[visible], parts_y[visible]
        part_ids_img = part_ids_img[visible]
        if not len(parts_x):
            return None, None

    # x1, y1, x2, y2 (K x 4)
    parts_bboxes = torch.stack([parts_x, parts_y, parts_x + 1, parts_y + 1], dim=1)
    return parts_bboxes, part_ids_img


def _preprocess_parts(parts):
    min_part, max_part = parts["part_id"].min(), parts["part_id"].max()
    num_parts = max_part + 1
    logger.info(
        f"Min. part_id={min_part}, max. part_id={max_part}. Assuming " f"that there are {num_parts} part IDs in total."
    )

    # Visible parts only!
    parts = parts[parts["visible"].astype(bool)]
    return parts, num_parts


def _pixel_space_bboxes(
    model: ProtoNet,
    x: torch.Tensor,  # N x C x H x W
    l1_radius: int,
    method: str,
):
    # get model results
    result = model(x)
    # max_sims: N x P
    min_dist_idxs = result["min_dist_idxs"]
    # proto_dists: N x P x Hz x Wz
    proto_dists = result["distances"]

    if method == "bbox":
        # bboxes: N x P x 4 (x1, y1, x2, y2)
        bboxes = model.pixel_space_bboxes(min_dist_idxs, proto_dists)
        bbox_centers_x = (bboxes[:, :, 0] + bboxes[:, :, 2]) / 2  # N x P
        bbox_centers_y = (bboxes[:, :, 1] + bboxes[:, :, 3]) / 2  # N x P

    elif method in {"heatmap_max", "heatmap_mean"}:
        bbox_centers_x = torch.empty(
            (
                len(x),
                model.num_prototypes,
            )
        )
        bbox_centers_y = torch.empty_like(bbox_centers_x)

        for i, (x_i, proto_dists_i) in enumerate(zip(x, proto_dists)):
            # P x H x W
            heatmap_max, heatmap_mean, _ = model.pixel_space_map(x_i, proto_dists_i)
            heatmap = heatmap_max if method == "heatmap_max" else heatmap_mean
            # P
            max_per_proto = heatmap.max(dim=2).values.max(dim=1).values
            # where the max is
            idx_p, idx_h, idx_w = torch.where(heatmap == max_per_proto[:, None, None])
            for j in range(model.num_prototypes):
                j_mask = idx_p == j
                bbox_centers_y[i, j] = idx_h[j_mask].float().mean()
                bbox_centers_x[i, j] = idx_w[j_mask].float().mean()

    elif method == "upsample":

        bbox_centers_x, bbox_centers_y = model.pixel_space_centers_upscale(x, min_dist_idxs, proto_dists)

    elif method == "center":

        bbox_centers_x = torch.full((len(x), model.num_prototypes), x.shape[3] // 2)
        bbox_centers_y = torch.full((len(x), model.num_prototypes), x.shape[2] // 2)

    else:
        raise NotImplementedError(method)
    zero = torch.tensor(0, dtype=bbox_centers_x.dtype)
    bboxes = torch.stack(
        [
            torch.maximum(bbox_centers_x - l1_radius, zero),
            torch.maximum(bbox_centers_y - l1_radius, zero),
            bbox_centers_x + l1_radius,
            bbox_centers_y + l1_radius,
        ],
        dim=2,
    )
    return bboxes


def _compute_intersects(
    bbox_i,  # P x 4
    parts_bboxes,  # K x 4
    class_mask,  # P
    part_ids_img,  # K
    soft: bool = False,
):
    if soft:
        raise NotImplementedError

    # P x 4, K x 4 --> P x K (for each visible part_id and proto)
    ious = box_iou(bbox_i[class_mask], parts_bboxes)
    # binarize it (convert to an intersection indicator)
    intersects = (ious > 0).int()  # P x K
    # adjust part_ids_img and intersects to handle the case of
    #  multiple instances of the same part in the image
    uniq_part_ids_img = np.unique(part_ids_img)
    if len(uniq_part_ids_img) != len(part_ids_img):
        # there are duplicates...
        # P x Kact
        intersects_dedupe = torch.zeros(
            (intersects.shape[0], len(uniq_part_ids_img)), dtype=intersects.dtype, device=intersects.device
        )
        # find maximum per actual
        for k, part in enumerate(uniq_part_ids_img):
            part_mask = part_ids_img == part
            # final intersect value is the maximum among all
            #  duplicate parts per prototype
            intersects_dedupe[:, k] = torch.max(
                intersects[:, part_mask],
                dim=1,
            ).values
        # finalize
        part_ids_img = uniq_part_ids_img
        intersects = intersects_dedupe
    # intersects:   P x K
    return part_ids_img, intersects


def consistency(  # https://arxiv.org/abs/2212.05946
    model: ProtoNet,
    data: DataLoader,  # test set, must contain img_id and x_raw
    parts: pd.DataFrame,
    config,
    metadata: Optional[pd.DataFrame],
    consistent_threshold: Optional[int] = 0.8,
    l1_radius: int = 36,  # 72-pixel diameter
    soft: bool = False,  # IoU instead of yes/no (different from final soft)
    method: str = "bbox",
):
    """"""
    if not model.class_specific:
        raise NotImplementedError

    prototype_class_identity = model.prototype_class_identity.cpu().bool()

    parts, num_parts = _preprocess_parts(parts)

    # Preprocess coordinate metadata
    dataset = config.dataset.name.upper().replace("-", "")
    crop_size, _ = pp_data.DATA_CROP_AND_PAD[dataset]
    resize = pp_data.DATA_RESIZE.get(dataset)
    downsample = pp_data.DATA_DOWNSAMPLE.get(dataset)
    bbox_cropped = "BBOX" in dataset

    with torch.no_grad():
        device = model.prototype_vectors.device

        contains_part_counts = torch.zeros((model.num_prototypes, num_parts))
        part_counts = torch.zeros_like(contains_part_counts)

        for i, (img_ids, orig_shapes, x, y) in enumerate(tqdm(data, desc="Samples")):
            x = x.to(device)
            y = y.to(device)

            # Get protonet-generated bboxes in pixel space
            bboxes = _pixel_space_bboxes(model, x, l1_radius, method)

            for j, (img_id, orig_shape, bbox_i, y_i) in enumerate(
                tqdm(zip(img_ids, orig_shapes, bboxes, y), desc="Batch", total=len(x), leave=(i + 1) == len(data))
            ):
                # preprocess part xy information
                parts_bboxes, part_ids_img = _preprocess_part_annotations_img(
                    orig_shape=orig_shape,
                    parts=parts,
                    metadata=metadata,
                    img_id=img_id,
                    crop_size=crop_size,
                    resize=resize,
                    downsample=downsample,
                    bbox_cropped=bbox_cropped,
                )
                if parts_bboxes is None:
                    continue

                # P x C --> P
                class_mask = prototype_class_identity[:, y_i]
                # intersects:   P x K
                part_ids_img, intersects = _compute_intersects(
                    bbox_i=bbox_i,  # P x 4
                    parts_bboxes=parts_bboxes,  # K x 4
                    class_mask=class_mask,  # P
                    part_ids_img=part_ids_img,  # K
                    soft=soft,
                )
                # update both counts for the visible parts
                class_mask_where = torch.where(class_mask)[0]
                contains_part_counts[
                    class_mask_where[:, None],  # Psub x 1
                    part_ids_img[None, :],  # 1 x Ksub
                ] += intersects
                part_counts[
                    class_mask_where[:, None],  # Psub x 1
                    part_ids_img[None, :],  # 1 x Ksub
                ] += 1

        # finalize scores
        scores_soft_all_parts = contains_part_counts.float() / part_counts.float()
        scores_soft_per_proto = torch.max(scores_soft_all_parts, dim=1).values

        # print some distribution stats
        q = torch.tensor([0, 0.25, 0.50, 0.75, 1])
        dist_percentiles = torch.quantile(scores_soft_per_proto[~torch.isnan(scores_soft_per_proto)], q).tolist()
        print(
            f"scores_soft_per_proto percentiles:\n"
            f'    {" / ".join(f"{x * 100:6.2f}%" for x in q.tolist())}\n'
            f'    {" / ".join(f"{x:7.4f}" for x in dist_percentiles)}'
        )

        score_soft = scores_soft_per_proto.nanmean()
        scores_hard_per_proto = (scores_soft_per_proto >= consistent_threshold).float()
        score_hard = scores_hard_per_proto.mean()

        return score_soft, score_hard


def stability(  # https://arxiv.org/abs/2212.05946
    model: ProtoNet,
    data: DataLoader,  # test set, must contain img_id and x_raw
    parts: pd.DataFrame,
    config,
    metadata: Optional[pd.DataFrame],
    l1_radius: int = 36,  # 72-pixel diameter
    noise_std: float = 0.2,
    min_overlap: Optional[float] = None,  # hard only
    soft: bool = False,  # IoU instead of yes/no (different from final soft)
    method: str = "bbox",
):
    """"""
    if not model.class_specific:
        raise NotImplementedError
    if min_overlap is not None:
        raise NotImplementedError

    prototype_class_identity = model.prototype_class_identity.cpu().bool()

    parts, num_parts = _preprocess_parts(parts)

    # Preprocess coordinate metadata
    dataset = config.dataset.name.upper().replace("-", "")
    crop_size, _ = pp_data.DATA_CROP_AND_PAD[dataset]
    resize = pp_data.DATA_RESIZE.get(dataset)
    downsample = pp_data.DATA_DOWNSAMPLE.get(dataset)
    bbox_cropped = "BBOX" in dataset

    with torch.no_grad():
        device = model.prototype_vectors.device

        stability_counts = torch.zeros((model.num_prototypes,))
        stability_ious = torch.zeros_like(stability_counts)
        prototype_counts = torch.zeros_like(stability_counts)

        for i, (img_ids, orig_shapes, x, y) in enumerate(tqdm(data, desc="Samples")):
            x = x.to(device)
            y = y.to(device)

            # Get protonet-generated bboxes in pixel space
            bboxes = _pixel_space_bboxes(model, x, l1_radius, method)
            noise = torch.zeros_like(x).normal_(mean=0, std=noise_std)
            x_noisy = x + noise
            bboxes_noisy = _pixel_space_bboxes(model, x_noisy, l1_radius, method)

            for j, (img_id, orig_shape, bbox_i, bbox_noisy_i, y_i) in enumerate(
                tqdm(
                    zip(img_ids, orig_shapes, bboxes, bboxes_noisy, y),
                    desc="Batch",
                    total=len(x),
                    leave=(i + 1) == len(data),
                )
            ):
                # preprocess part xy information
                parts_bboxes, part_ids_img = _preprocess_part_annotations_img(
                    orig_shape=orig_shape,
                    parts=parts,
                    metadata=metadata,
                    img_id=img_id,
                    crop_size=crop_size,
                    resize=resize,
                    downsample=downsample,
                    bbox_cropped=bbox_cropped,
                )
                if parts_bboxes is None:
                    continue

                # P x C --> P
                class_mask = prototype_class_identity[:, y_i]
                # intersects:   Psub x Kact
                _, intersects = _compute_intersects(
                    bbox_i=bbox_i,  # P x 4
                    parts_bboxes=parts_bboxes,  # K x 4
                    class_mask=class_mask,  # P
                    part_ids_img=part_ids_img,  # K
                    soft=soft,
                )
                _, intersects_noisy = _compute_intersects(
                    bbox_i=bbox_noisy_i,  # P x 4
                    parts_bboxes=parts_bboxes,  # K x 4
                    class_mask=class_mask,  # P
                    part_ids_img=part_ids_img,  # K
                    soft=soft,
                )

                # test for equivalence before and after adding noise (same
                #  object assignments per prototype)
                # P
                same_intersects = (intersects == intersects_noisy).all(dim=1)
                intersect_ious = (intersects & intersects_noisy).sum(dim=1) / (intersects | intersects_noisy).sum(dim=1)

                # update both counts for stability indicator and proto count
                stability_counts[class_mask] += same_intersects.int()
                stability_ious[class_mask] += intersect_ious
                prototype_counts[class_mask] += 1

        # finalize scores
        scores_per_proto = stability_counts / prototype_counts
        soft_scores_per_proto = stability_ious / prototype_counts

        # print some distribution stats
        q = torch.tensor([0, 0.25, 0.50, 0.75, 1])
        dist_percentiles = torch.quantile(scores_per_proto[~torch.isnan(scores_per_proto)], q).tolist()
        print(
            f"scores_per_proto percentiles:\n"
            f'    {" / ".join(f"{x * 100:6.2f}%" for x in q.tolist())}\n'
            f'    {" / ".join(f"{x:7.4f}" for x in dist_percentiles)}'
        )
        dist_percentiles = torch.quantile(soft_scores_per_proto[~torch.isnan(soft_scores_per_proto)], q).tolist()
        print(
            f"soft_scores_per_proto percentiles:\n"
            f'    {" / ".join(f"{x * 100:6.2f}%" for x in q.tolist())}\n'
            f'    {" / ".join(f"{x:7.4f}" for x in dist_percentiles)}'
        )

        score = scores_per_proto.nanmean()
        score_soft = soft_scores_per_proto.nanmean()
        return score_soft, score


def relevance_ordering_test(
    model: ProtoNet,
    data: DataLoader,  # test set
    num_samples: int = 50,
    same_class: bool = True,  # prototypes of same class only
    normalized: bool = False,
    prop_pixels: float = 0.5,
    method: str = "rf",
    zeros=True,  # if false, use random
    seed: Optional[int] = None,
    config: Optional[object] = None,
    savedir_for_viz: Optional[str] = None,
):
    assert 0 < prop_pixels <= 1

    if same_class:
        if not model.class_specific:
            raise ValueError
        prototype_class_identity = model.prototype_class_identity.cpu().bool()

    with torch.no_grad():
        device = model.prototype_vectors.device
        batch_size = data.batch_size

        rand_sample = None
        num_pixels = None
        cum_sims = None
        num_samples_eval = 0
        num_samples_eval_per_proto = torch.zeros((model.num_prototypes,))

        pbar = tqdm(data, total=num_samples, desc="Samples")
        for i, (x, y) in enumerate(pbar):
            if num_samples_eval == num_samples:
                break
            x = x.to(device)
            # truncate to avoid evaluating more samples than needed
            x = x[: num_samples - num_samples_eval]
            if rand_sample is None:
                N, C, H, W = x.shape
                # assumes normally distributed data w/ mean=0 std=1
                if seed is not None:
                    torch.manual_seed(seed)
                if zeros:
                    rand_sample = torch.zeros((1, C, H, W), device=device, dtype=x.dtype)
                else:
                    rand_sample = torch.randn((1, C, H, W), device=device, dtype=x.dtype)
                # +1 for random without any pixels added back
                num_pixels = round(H * W * prop_pixels) + 1
                # cumulative sims tensor
                cum_sims = torch.zeros((model.num_prototypes, num_pixels), dtype=x.dtype, device=device)

            result = model(x)
            # proto_dists: N x P x Hz x Wz
            proto_dists = result["distances"]
            # max_sims: N x P
            max_sims = result["max_similarities"]

            leave = ((i + 1) * len(x)) == num_samples
            for k, (x_i, y_i, proto_dists_i, max_sims_i) in enumerate(
                tqdm(zip(x, y, proto_dists, max_sims), desc="Batch", total=len(x), leave=leave)
            ):
                # The pixel-space similarity map for each prototype
                # P x H x W
                if method == "rf":
                    heat_map_max, heat_map_avg, heat_map_sum = model.pixel_space_map(x_i, proto_dists_i)
                    heat_map = heat_map_max
                elif method == "upscale":
                    heat_map = model.pixel_space_upscale(x_i, proto_dists_i)
                elif method == "random":
                    heat_map = torch.rand(model.num_prototypes, H, W)
                elif method == "prp":
                    x_i_var = Variable(x_i.unsqueeze(0))
                    x_i_var = x_i_var.to(device)

                    heat_map = []

                    for p in range(model.num_prototypes):
                        heat_map_p = generate_prp_image(x_i_var, p, model, config)
                        heat_map.append(heat_map_p)
                    heat_map = torch.stack(heat_map)
                else:
                    raise NotImplementedError

                # Get descending sort indices: P x HW
                heat_map_sort_idxs = torch.argsort(heat_map.reshape(model.num_prototypes, -1), descending=True)

                if same_class:
                    proto_idxs = torch.where(prototype_class_identity[:, y_i])[0]
                else:
                    proto_idxs = range(model.num_prototypes)

                # For each prototype
                leave_proto = (k + 1) == len(x)
                for m, p in enumerate(tqdm(proto_idxs, desc="Prototypes", leave=leave and leave_proto)):
                    if savedir_for_viz:
                        sample_sims = torch.zeros((num_pixels,), dtype=x.dtype, device=device)
                    # N x C x H x W
                    x_pert = rand_sample.repeat(batch_size, 1, 1, 1)

                    num_pixels_eval = 0
                    # Batch evaluate each pixel added back in descending
                    #  pixel-space similarity score order
                    pbar_pixels = tqdm(
                        desc="Pixels", total=num_pixels, leave=(leave and leave_proto and ((m + 1) == len(proto_idxs)))
                    )
                    while num_pixels_eval < num_pixels:
                        # truncate to avoid duplicate evaluations
                        x_pert = x_pert[: num_pixels - num_pixels_eval]

                        # create a batch of new patches
                        for j in range(len(x_pert)):
                            if j == 0 and num_pixels_eval == 0:
                                # the very first of `num_pixels` is allocated to
                                #  just the random image
                                continue
                            # update batch (current samples and all proceeding
                            #  samples)
                            pixel_idx = heat_map_sort_idxs[p, j + num_pixels_eval - 1]  # -1 for all rand
                            pixel_idx_h, pixel_idx_w = unravel_index(pixel_idx, heat_map.shape[1:])
                            x_pert[j:, :, pixel_idx_h, pixel_idx_w] = x_i[None, :, pixel_idx_h, pixel_idx_w]

                        # eval batch, get max_sims for prototype p
                        sims = model(x_pert)["max_similarities"][:, p]  # N
                        if normalized:
                            # sample-wise and prototype-wise max normalization
                            sims /= max_sims_i[p]
                        # update cumulative sims for this prototype and the
                        #  current evaluated pixels
                        # @formatter:off
                        cum_sims[p, num_pixels_eval : num_pixels_eval + len(sims)] += sims
                        if savedir_for_viz:
                            sample_sims[num_pixels_eval : num_pixels_eval + len(sims)] = sims
                        # @formatter:on

                        # at end, update number of evaluated pixels
                        num_pixels_eval += len(x_pert)
                        # and update all samples to be at the value of final
                        #  pixel added
                        x_pert[:] = x_pert[-1][None]

                        pbar_pixels.update(len(x_pert))

                    pbar_pixels.close()

                    if savedir_for_viz:
                        sample_idx = num_samples_eval + k
                        savedir_full = os.path.join(
                            savedir_for_viz,
                            f"method_{method}",
                            f"sample_{sample_idx}",
                            f"prototype_{int(p)}",
                        )
                        os.makedirs(savedir_full, exist_ok=True)
                        # Save
                        torch.save(
                            {
                                "similarities": sample_sims.cpu(),
                                "random_sample": rand_sample.cpu(),
                                "heat_map_indices": heat_map_sort_idxs[p].cpu(),
                            },
                            os.path.join(savedir_full, "rot_data.pt"),
                        )

            num_samples_eval += len(x)
            num_samples_eval_per_proto[proto_idxs] += len(x)
            pbar.update(len(x))

        pbar.close()

        cum_sims = cum_sims.cpu()
        cum_sims /= num_samples_eval_per_proto[:, None]
        # P x num_pixels --> num_pixels
        cum_sims_agg = torch.nanmean(cum_sims, dim=0)

        return cum_sims, cum_sims_agg
