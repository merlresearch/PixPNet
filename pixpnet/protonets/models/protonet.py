# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import warnings
from collections import OrderedDict
from math import ceil
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pixpnet.protonets.models.feature_extractor_zoo import get_feature_extractor
from pixpnet.protonets.models.layers import GroupedLinear, GroupedSum
from pixpnet.symbolic.models import compute_rf_data
from pixpnet.utils import get_logger, log_once
from pixpnet.utils_torch import take_rf_from_bbox, unravel_index

logger = get_logger(__name__)


def compute_distances(metric, x, prototype_vectors, ones=None):
    if metric == "l2":
        return l2_convolution(x, prototype_vectors, ones)
    elif metric == "cosine":
        return cosine_convolution(x, prototype_vectors)
    else:
        raise NotImplementedError(metric)


def l2_convolution(x, prototype_vectors, ones):
    """
    Apply prototype_vectors as l2-convolution filters on input x and
    compute the L2 norm of x-p.

    ||a - b||^2_2
        = (a - b)^2
        = a^2 - 2ab + b^2
    In this context:
        = x2_patch_sum - 2 * xp + p2
        = x2_patch_sum - (xp + xp) + p2
    ReLU is used to ensure a positive distance (just in case™).
    """
    # P: num_prototypes (e.g. 2000)
    # D: prototype dimensionality (e.g. 512)
    # K: prototype_kernel_size (1 in paper)  noqa: E800
    # input:        N x D x H x W
    # weight:       P x D x K x K
    # x2_patch_sum: N x P x H-K+1 x W-K+1 (N x P x H x W when K=1)
    x2_patch_sum = F.conv2d(input=x * x, weight=ones)

    # p2 is a vector of shape (P,) reshaped to (P, 1, 1)
    p2 = torch.sum(prototype_vectors * prototype_vectors, dim=(1, 2, 3)).view(-1, 1, 1)

    # input:        N x D x H x W
    # weight:       P x D x K x K
    # xp:           N x P x H-K+1 x W-K+1 (N x P x H x W when K=1)
    # distances:    same shape as xp
    xp = F.conv2d(input=x, weight=prototype_vectors)
    distances = F.relu(x2_patch_sum - (xp + xp) + p2)

    return distances


def cosine_convolution(x, prototype_vectors):
    # An alternative distance metric used in TesNet. Alternative to
    #  l2_convolution
    x = F.normalize(x, p=2, dim=1)
    prototype_vectors = F.normalize(prototype_vectors, p=2, dim=1)
    similarities = F.conv2d(input=x, weight=prototype_vectors)
    # clip similarities in the range [-1, +1] (numerical error can
    #  cause similarities to be outside this range)
    similarities = torch.clamp(similarities, -1, 1)
    distances = 1 - similarities  # bounded [0, 2]
    return distances


def gaussian_kernel(size, sigma=1.0, device=None):
    """
    creates a pseudo gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = torch.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size, device=device)
    gauss = torch.exp(-0.5 * torch.square(ax) / (sigma**2))
    kernel = torch.outer(gauss, gauss)
    return kernel / torch.max(kernel)  # max-normalize (max value is 1)


SlicesType = List[List[List[Tuple[slice, slice]]]]


class ProtoNet(nn.Module):
    # Buffers
    ones: torch.Tensor
    corresponding_sample_idxs: torch.Tensor
    min_fmap_idxs: torch.Tensor
    prototype_class_identity: Optional[torch.Tensor]
    # Parameters
    prototype_vectors: torch.nn.Parameter

    # Constants
    prototype_layer_stride = 1

    def __init__(
        self,
        features: nn.Module,
        feature_layer: str,
        rf_slices: Optional[SlicesType],
        num_prototypes: int,
        prototype_dim: int,
        prototype_kernel_size: int,
        num_classes: int,
        init_weights: bool = True,
        prototype_activation: Union[str, Callable] = "log",
        add_on_layers_type: str = "regular",
        class_specific: bool = True,
        epsilon: float = 1e-6,
        learn_prototypes: bool = True,
        incorrect_strength: float = -0.5,
        correct_strength: float = 1,
        readout_type: str = "linear",
        distance: str = "l2",
    ):
        """"""
        super().__init__()
        self.prototype_shape = (num_prototypes, prototype_dim, prototype_kernel_size, prototype_kernel_size)
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        self.prototype_kernel_size = prototype_kernel_size
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.learn_prototypes = learn_prototypes
        # prototype_activation could be 'log', 'linear',
        #  or a callable that converts distance to similarity score
        self.prototype_activation = prototype_activation
        self.distance = distance
        self.feature_layer = feature_layer

        self.rf_slices = rf_slices
        self.rf_idxs = None
        self.rf_sizes = None
        if self.rf_slices is not None:
            Hz = len(self.rf_slices)
            Wz = len(self.rf_slices[0])
            self.rf_sizes = torch.zeros((Hz, Wz, 2), dtype=torch.int)
            self.rf_idxs = torch.zeros((Hz, Wz, 4), dtype=torch.int)
            for h in range(Hz):
                for w in range(Wz):
                    # for patch h,w
                    if len(self.rf_slices[h][w]) > 1:
                        raise NotImplementedError
                    for h_s, w_s in self.rf_slices[h][w]:
                        # Start weighting approach
                        h_size = h_s.stop - h_s.start
                        w_size = w_s.stop - w_s.start
                        self.rf_sizes[h, w] = torch.tensor([h_size, w_size], dtype=torch.int)
                        self.rf_idxs[h, w] = torch.tensor([h_s.start, h_s.stop, w_s.start, w_s.stop], dtype=torch.int)

        self.incorrect_strength = incorrect_strength
        self.correct_strength = correct_strength
        self.class_specific = class_specific
        if self.class_specific:
            # Here we are initializing the class identities of the prototypes.
            # Without domain specific knowledge we allocate the same number of
            # prototypes for each class
            assert self.num_prototypes % self.num_classes == 0
            # a one-hot indication matrix for each prototype's class identity
            self.register_buffer(
                "prototype_class_identity", torch.zeros(self.num_prototypes, self.num_classes, dtype=torch.int)
            )
            num_prototypes_per_class = self.num_prototypes // self.num_classes
            for j in range(self.num_prototypes):
                self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        # this has to be named features to allow the precise loading
        self.features = features
        self._init_add_on_layers(add_on_layers_type)

        self.register_parameter(
            "prototype_vectors", nn.Parameter(torch.rand(self.prototype_shape), requires_grad=learn_prototypes)
        )
        self.register_buffer("ones", torch.ones(self.prototype_shape))
        self.register_buffer("corresponding_sample_idxs", torch.full((self.num_prototypes,), -1))
        self.register_buffer("min_fmap_idxs", torch.full((self.num_prototypes, 4), -1))

        self.readout_type = readout_type
        self._init_last_layer()

        if init_weights:
            self._initialize_weights()

    def _init_last_layer(self):
        # do not use bias to aid interpretability
        if self.readout_type == "linear":  # standard linear
            self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)
        elif self.readout_type == "sparse":  # sparse linear
            if not self.class_specific:
                raise ValueError('`readout_type` cannot be "sparse" if ' "`class_specific` is False")
            self.last_layer = GroupedLinear(self.num_prototypes, self.num_classes, groups=self.num_classes, bias=False)
        elif self.readout_type == "proto":  # prototype sim sums as prediction
            if not self.class_specific:
                raise ValueError('`readout_type` cannot be "proto" if ' "`class_specific` is False")
            # Note that this assumes that `prototype_class_identity` is still
            #  uniform across classes when class_specific is True
            self.last_layer = GroupedSum(self.num_prototypes, self.num_classes)
        else:
            raise NotImplementedError(f"readout_type = {self.readout_type}")

    def _init_add_on_layers(self, add_on_layers_type):
        in_channels = self.features.out_channels

        final_act, final_act_str = nn.Sigmoid(), "sigmoid"
        if add_on_layers_type == "bottleneck":
            add_on_layers = []
            current_in_channels = in_channels
            conv_idx = 1
            while current_in_channels > self.prototype_dim or not len(add_on_layers):
                current_out_channels = max(self.prototype_dim, (current_in_channels // 2))
                if current_out_channels > self.prototype_dim:
                    conv2_str, act2, act2_str = (f"conv{conv_idx + 1}", nn.ReLU(), f"relu{conv_idx + 1}")
                else:
                    assert current_out_channels == self.prototype_dim
                    conv2_str, act2, act2_str = ("conv_last", final_act, final_act_str)
                add_on_layers.extend(
                    (
                        (
                            f"conv{conv_idx}",
                            nn.Conv2d(
                                in_channels=current_in_channels, out_channels=current_out_channels, kernel_size=1
                            ),
                        ),
                        (f"relu{conv_idx}", nn.ReLU()),
                        (
                            conv2_str,
                            nn.Conv2d(
                                in_channels=current_out_channels, out_channels=current_out_channels, kernel_size=1
                            ),
                        ),
                        (act2_str, act2),
                    )
                )
                current_in_channels = current_in_channels // 2
                conv_idx += 2
        elif add_on_layers_type == "regular":
            add_on_layers = (
                ("conv1", nn.Conv2d(in_channels=in_channels, out_channels=self.prototype_dim, kernel_size=1)),
                ("relu1", nn.ReLU()),
                (
                    "conv_last",
                    nn.Conv2d(in_channels=self.prototype_dim, out_channels=self.prototype_dim, kernel_size=1),
                ),
                (final_act_str, final_act),
            )
        else:
            raise ValueError(add_on_layers_type)
        add_on_layers = OrderedDict(add_on_layers)

        self.add_on_layers = nn.Sequential(add_on_layers)

    def conv_features(self, x):
        """
        the feature input to prototype layer
        """
        x = self.features(x)
        log_once(logger.info, f'features output shape: {("N", *x.size()[1:])}')
        x = self.add_on_layers(x)
        log_once(logger.info, f'add_on_layers output shape: {("N", *x.size()[1:])}')
        return x

    def compute_distances(self, x):
        return compute_distances(self.distance, x, self.prototype_vectors, self.ones)

    def prototype_distances(self, x):
        """
        x is the raw input
        """
        conv_features = self.conv_features(x)
        distances = self.compute_distances(conv_features)
        return conv_features, distances

    def dist_2_sim(self, distances):
        if self.prototype_activation == "log":
            # equivalent:
            #  log((distances + 1) / (distances + epsilon))  # noqa: E800
            # but this one is numerically more accurate
            return torch.log(1 / (distances + self.epsilon) + 1)
        elif self.prototype_activation == "linear":
            if self.distance == "cosine":
                # dists = 1 - sim --> sim = 1 - dists
                return 1 - distances
            else:
                return -distances
        else:
            return self.prototype_activation(distances)

    def forward(self, x, return_features=False):
        result = self.prototype_distances(x)
        conv_features, distances = result
        outputs = self.classify_head(x, distances)
        if return_features:
            outputs["features"] = conv_features
        return outputs

    def classify_head(self, x, distances):
        return self._classify_head_proto2patch(distances)

    def pixel_space_map(self, x_i, proto_dists, sigma_factor=1.0):
        # Note: one sample at a time! otherwise there will definitely be
        #  memory issues on most hardware and ProtoNets
        dtype = proto_dists.dtype
        device = proto_dists.device

        # validate shape
        if x_i.ndim == 4:
            assert x_i.shape[0] == 1, x_i.shape
            x_i = torch.squeeze(x_i, 0)
        else:
            assert x_i.ndim == 3, x_i.shape

        if proto_dists.ndim == 4:
            assert proto_dists.shape[0] == 1, proto_dists.shape
            proto_dists = torch.squeeze(proto_dists, 0)
        else:
            assert proto_dists.ndim == 3, proto_dists.shape

        C, H, W = x_i.shape
        P, Hz, Wz = proto_dists.shape

        # dists --> sims
        proto_sims = self.dist_2_sim(proto_dists)
        # Sim maps
        heat_map_max = torch.zeros((P, H, W), dtype=dtype, device=device)
        heat_map_avg = torch.zeros_like(heat_map_max)
        heat_map_counts = torch.zeros_like(heat_map_avg, dtype=torch.int)

        rf_h = self.rf_sizes[:, :, 0].max()
        rf_w = self.rf_sizes[:, :, 1].max()

        do_super_rfs = rf_h >= H or rf_w >= W
        if do_super_rfs:
            # increase true rf_h/w
            where_big = torch.where((self.rf_sizes[:, :, 0] >= H) | (self.rf_sizes[:, :, 1] >= W))
            do_super_rfs = len(where_big[0]) > 1
            if do_super_rfs:
                # linear stretching assumption for super-100% RF networks
                naive_midpoints_h = torch.round((torch.arange(Hz) + 0.5) * H / Hz).int()
                naive_midpoints_w = torch.round((torch.arange(Wz) + 0.5) * W / Wz).int()

                im_midpoints = (H - 1) / 2, (W - 1) / 2

                pad_h = torch.round((im_midpoints[0] - naive_midpoints_h[where_big[0]]).abs().max()).int()
                pad_w = torch.round((im_midpoints[1] - naive_midpoints_w[where_big[1]]).abs().max()).int()

                # increase the RFs by the discovered padding amount
                rf_h = rf_h + 2 * pad_h
                rf_w = rf_w + 2 * pad_w

        k_size = max(rf_h, rf_w)
        sigma = k_size * sigma_factor
        g_kern = gaussian_kernel(k_size, sigma=sigma, device=device)

        for h in range(Hz):
            for w in range(Wz):
                # for patch h,w
                sims_hw = proto_sims[:, h, w][:, None, None]  # P x 1 x 1
                h_size, w_size = self.rf_sizes[h, w]  # rf_sizes: Hz x Wz x 2

                hs0, hs1, ws0, ws1 = self.rf_idxs[h, w]

                if do_super_rfs:
                    mh, mw = naive_midpoints_h[h], naive_midpoints_w[w]

                    hs0_ = mh - rf_h // 2
                    hs1_ = mh + ceil(rf_h // 2)
                    ws0_ = mw - rf_w // 2
                    ws1_ = mw + ceil(rf_w // 2)

                    h_pad0 = max(-hs0_, 0)
                    h_pad1 = max(hs1_ - H - max(hs0_, 0), 0)
                    w_pad0 = max(-ws0_, 0)
                    w_pad1 = max(ws1_ - W - max(ws0_, 0), 0)

                    if h_size < H:
                        if hs0 != 0:
                            h_pad0 += H - h_size
                        else:
                            h_pad1 += H - h_size
                    if w_size < W:
                        if ws0 != 0:
                            w_pad0 += W - w_size
                        else:
                            w_pad1 += W - w_size

                    g_kern_hw = g_kern[int(h_pad0) : k_size - ceil(h_pad1), int(w_pad0) : k_size - ceil(w_pad1)]
                else:
                    h_pad0 = h_pad1 = 0
                    w_pad0 = w_pad1 = 0
                    if h_size < rf_h:
                        if hs1 - rf_h < 0:
                            h_pad0 += rf_h - h_size
                        else:
                            h_pad1 += rf_h - h_size
                    if w_size < rf_w:
                        if ws1 - rf_w < 0:
                            w_pad0 += rf_w - w_size
                        else:
                            w_pad1 += rf_w - w_size
                    g_kern_hw = g_kern[int(h_pad0) : k_size - ceil(h_pad1), int(w_pad0) : k_size - ceil(w_pad1)]

                sims_hw_full = sims_hw * g_kern_hw[None, :, :]

                heat_map_avg[:, hs0:hs1, ws0:ws1] += sims_hw_full
                heat_map_counts[:, hs0:hs1, ws0:ws1] += 1
                heat_map_max[:, hs0:hs1, ws0:ws1] = torch.maximum(sims_hw_full, heat_map_max[:, hs0:hs1, ws0:ws1])
        # take element-wise averages according to overlap tensor (counts)
        heat_map_sum = heat_map_avg.clone()
        heat_map_avg /= heat_map_counts

        return heat_map_max, heat_map_avg, heat_map_sum  # each is P x H x W

    def pixel_space_upscale(self, x_i, proto_dists):
        # validate shape
        if x_i.ndim == 4:
            assert x_i.shape[0] == 1, x_i.shape
            x_i = torch.squeeze(x_i, 0)
        else:
            assert x_i.ndim == 3, x_i.shape

        if proto_dists.ndim == 4:
            assert proto_dists.shape[0] == 1, proto_dists.shape
            proto_dists = torch.squeeze(proto_dists, 0)
        else:
            assert proto_dists.ndim == 3, proto_dists.shape

        C, H, W = x_i.shape

        # dists --> sims
        proto_sims = self.dist_2_sim(proto_dists)
        # Sim maps
        heat_map = torch.nn.functional.interpolate(proto_sims[None], (H, W), mode="bicubic")
        # 1 x P x H x W --> P x H x W
        heat_map = heat_map.squeeze(dim=0)

        return heat_map

    def pixel_space_bboxes(self, min_dist_idxs, proto_dists):
        if not (self.prototype_kernel_size == self.prototype_layer_stride == 1):
            raise NotImplementedError((self.prototype_kernel_size, self.prototype_layer_stride))
        N, P = min_dist_idxs.shape
        # N x P, N x P
        fmap_h_start, fmap_w_start = unravel_index(min_dist_idxs, proto_dists.shape[-2:])

        bboxes = []
        for i in range(N):
            bboxes_i = []
            for j in range(P):
                h, w = fmap_h_start[i, j], fmap_w_start[i, j]
                slices_hw = self.rf_slices[h][w]
                assert len(slices_hw) == 1, "unsupported at the moment"
                slice_h, slice_w = slices_hw[0]
                x1, y1 = slice_w.start, slice_h.start
                x2, y2 = slice_w.stop, slice_h.stop
                bboxes_i.append([x1, y1, x2, y2])
            bboxes.append(bboxes_i)
        bboxes = torch.tensor(bboxes)
        return bboxes  # N x P x 4

    def pixel_space_centers_upscale(self, x, min_dist_idxs, proto_dists):
        if not (self.prototype_kernel_size == self.prototype_layer_stride == 1):
            raise NotImplementedError((self.prototype_kernel_size, self.prototype_layer_stride))
        _, _, H, W = x.shape
        Hz, Wz = proto_dists.shape[-2:]
        # N x P, N x P
        fmap_h_start, fmap_w_start = unravel_index(min_dist_idxs, [Hz, Wz])

        naive_midpoints_h = torch.round((torch.arange(Hz) + 0.5) * H / Hz).int()
        naive_midpoints_w = torch.round((torch.arange(Wz) + 0.5) * W / Wz).int()

        centers_x = naive_midpoints_w[fmap_w_start.cpu()]
        centers_y = naive_midpoints_h[fmap_h_start.cpu()]

        return centers_x, centers_y  # NxP each

    def _classify_head_proto2patch(self, distances):
        # global min pooling (N x P x H x W --> N x P x 1 x 1)
        # I.e., the KxK patch of the latent representations z of the input
        # images that is most similar to each of the P prototypes. Output
        # indicates how present each prototype is in the image.
        min_distances, min_dist_idxs = self.global_min_pool(distances)
        # Convert distances to similarity using the log/linear function
        prototype_activations = self.dist_2_sim(min_distances)

        # Compute logits (N x C)
        logits = self.last_layer(prototype_activations)

        return {
            "logits": logits,  # N x C
            "min_distances": min_distances,  # N x P
            "min_dist_idxs": min_dist_idxs,  # N x P
            "distances": distances,  # N x P x H x W
            "max_similarities": prototype_activations,  # N x P
        }

    @staticmethod
    def global_min_pool(distances):
        """
        To gather `min_distances` using `min_dist_idxs`:

        ```python
        distances.flatten(start_dim=2).gather(
            dim=2, index=min_dist_idxs.flatten(start_dim=2)
        ).view_as(min_dist_idxs)
        ```

        :param distances:
        :return:
        """
        with warnings.catch_warnings():
            # You'd think they would've checked for positionally passed args...
            warnings.filterwarnings(
                "ignore", ".*order of the arguments: ceil_mode and " "return_indices will change.*", UserWarning
            )
            min_distances, min_dist_idxs = F.max_pool2d(
                -distances, kernel_size=(distances.size()[2], distances.size()[3]), return_indices=True
            )
        min_distances = -min_distances
        # N x P x 1 x 1 --> N x P
        min_distances = min_distances.view(min_distances.shape[0], min_distances.shape[1])
        min_dist_idxs = min_dist_idxs.view(min_dist_idxs.shape[0], min_dist_idxs.shape[1])
        return min_distances, min_dist_idxs

    def push_forward(self, x):
        """this method is needed for the pushing operation"""
        return self.prototype_distances(x)

    def set_prototypes(self, new_prototype_vectors, corresponding_sample_idxs=None, min_fmap_idxs=None):
        self.prototype_vectors.data.copy_(new_prototype_vectors)
        err_msg = "both min_fmap_idxs and corresponding_sample_idxs should be" " None or not None"
        if corresponding_sample_idxs is not None:
            assert min_fmap_idxs is not None, err_msg
            self.corresponding_sample_idxs = corresponding_sample_idxs
            self.min_fmap_idxs = min_fmap_idxs
        else:
            assert min_fmap_idxs is None, err_msg

    def prune_prototypes(self, prototypes_to_prune):
        """
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        """
        prototypes_to_keep = [*({*range(self.num_prototypes)} - {*prototypes_to_prune})]

        self.register_parameter(
            "prototype_vectors",
            nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...], requires_grad=self.learn_prototypes),
        )
        self.corresponding_sample_idxs = self.corresponding_sample_idxs[prototypes_to_keep, ...]
        self.min_fmap_idxs = self.min_fmap_idxs[prototypes_to_keep, ...]

        self.prototype_shape = tuple(self.prototype_vectors.size())
        self.num_prototypes = self.prototype_shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are
        # consistent
        if self.readout_type != "linear":
            raise NotImplementedError(
                f"Removing prototypes for readout_type={self.readout_type}" f" is not implemented yet"
            )
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        # self.ones is nn.Parameter
        self.ones = self.ones[prototypes_to_keep, ...]

        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        if self.class_specific:
            self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def set_last_layer_incorrect_connection(self):
        """
        Initialize weight of last_layer to correct_strength if
        prototype_class_identity is 1 (i.e., the prototype is for that class),
        and to incorrect_strength if prototype_class_identity is 0 (i.e., the
        prototype is not for that class)
        """
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        self.last_layer.weight.data.copy_(
            self.correct_strength * positive_one_weights_locations
            + self.incorrect_strength * negative_one_weights_locations
        )

    def _initialize_weights(self):
        for name, m in self.add_on_layers.named_children():
            if isinstance(m, nn.Conv2d):
                if name == "conv_last":
                    # for the sigmoid activation
                    nn.init.xavier_normal_(m.weight, gain=1.0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.class_specific and self.readout_type == "linear":
            # This is not needed (or valid) for sparse linear or proto
            self.set_last_layer_incorrect_connection()
        elif self.class_specific and self.readout_type == "sparse":
            nn.init.ones_(self.last_layer.weight)


def protonet(
    feature_extractor,
    feature_layer=None,
    pretrained=True,
    num_prototypes=2000,
    prototype_dim=512,
    prototype_kernel_size=1,
    num_classes=200,
    input_size=224,
    init_weights=True,
    prototype_activation: Union[str, Callable] = "log",
    add_on_layers_type="regular",
    class_specific=True,
    epsilon=1e-6,
    learn_prototypes=True,
    incorrect_strength=-0.5,
    correct_strength=1,
    readout_type="linear",
    distance="l2",
):
    """"""
    if isinstance(feature_extractor, str):
        last_module_name = []
        if feature_layer:
            last_module_name.append(feature_layer)
        if len(last_module_name) == 1:
            last_module_name = last_module_name[0]
        features = get_feature_extractor(
            feature_extractor,
            pretrained=pretrained,
            last_module_name=last_module_name or None,
        )
        _, rf_data = compute_rf_data(feature_extractor, input_size, input_size, num_classes=1)
        rf_layer = rf_data[feature_layer]
        h_z, w_z = rf_layer.shape[-2:]
        rf_slices = []
        for h in range(h_z):
            slices_h = []
            for w in range(w_z):
                rf_feat_hw = take_rf_from_bbox(
                    rf_layer, h, w, prototype_kernel_size, prototype_kernel_size, ProtoNet.prototype_layer_stride
                )
                slices_hw = []
                for slice_hw in rf_feat_hw.as_slices(all_channels=True):
                    _, _, h_s, w_s = slice_hw
                    slices_hw.append((h_s, w_s))
                slices_h.append(slices_hw)
            rf_slices.append(slices_h)

    else:
        features = feature_extractor
        rf_slices = None

    if feature_layer is None:
        feature_layer = features.last_module_name[0] if features.multi_output else features.last_module_name

    return ProtoNet(
        features=features,
        feature_layer=feature_layer,
        rf_slices=rf_slices,
        num_prototypes=num_prototypes,
        prototype_dim=prototype_dim,
        prototype_kernel_size=prototype_kernel_size,
        num_classes=num_classes,
        init_weights=init_weights,
        prototype_activation=prototype_activation,
        add_on_layers_type=add_on_layers_type,
        class_specific=class_specific,
        epsilon=epsilon,
        learn_prototypes=learn_prototypes,
        incorrect_strength=incorrect_strength,
        correct_strength=correct_strength,
        readout_type=readout_type,
        distance=distance,
    )
