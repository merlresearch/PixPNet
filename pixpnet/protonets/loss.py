# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
from torch import Tensor, nn

from pixpnet.protonets.models.layers import GroupedLinear
from pixpnet.protonets.models.protonet import ProtoNet


class ClusterLoss(nn.Module):
    def __init__(self, class_specific=True):
        super().__init__()
        self.class_specific = class_specific

    def forward(self, min_distances: Tensor, target: Tensor, model: ProtoNet) -> Tensor:
        # min_distances:       N x P
        if self.class_specific:
            # prototypes_of_correct_class: batch_size x num_prototypes
            prototypes_of_correct_class = torch.t(model.prototype_class_identity[:, target])
            min_distances_target = torch.where(
                prototypes_of_correct_class.bool(),
                min_distances,
                torch.tensor(torch.inf, dtype=min_distances.dtype, device=min_distances.device),
            )
            min_min_distances, _ = torch.min(min_distances_target, dim=1)
            cluster_loss = torch.mean(min_min_distances)
        else:
            min_min_distances, _ = torch.min(min_distances, dim=1)
            cluster_loss = torch.mean(min_min_distances)

        return cluster_loss


class SeparationLoss(nn.Module):
    @staticmethod
    def forward(min_distances: Tensor, target: Tensor, model: ProtoNet, return_avg: bool = False):
        """
        Here we want to maximize the minimum of all minimum proto-patch
        distances (each being some patch that is closest to a given prototype)
        for each non-class prototype. In effect, for each sample, a patch is
        selected for each non-class prototype according to minimum distance. So,
        we end up with one patch and one prototype per sample after taking the
        minimum of the proto-patch distances.
        """
        # min_distances: N x P
        # prototype_class_identity: P x C
        # prototypes_of_correct_class: N x P
        prototypes_of_correct_class = torch.t(model.prototype_class_identity[:, target]).bool()
        min_distances_nontarget = torch.where(
            prototypes_of_correct_class.bool(),
            torch.tensor(torch.inf, dtype=min_distances.dtype, device=min_distances.device),
            min_distances,
        )
        dists_to_nontarget_prototypes, _ = torch.min(min_distances_nontarget, dim=1)
        separation_loss = -torch.mean(dists_to_nontarget_prototypes)

        if not return_avg:
            return separation_loss
        # otherwise
        min_distances_nontarget = torch.where(
            prototypes_of_correct_class.bool(),
            torch.tensor(0, dtype=min_distances.dtype, device=min_distances.device),
            min_distances,
        )
        avg_separation_cost = torch.sum(min_distances_nontarget, dim=1) / torch.sum(
            ~prototypes_of_correct_class.bool(), dim=1
        )
        avg_separation_cost = -torch.mean(avg_separation_cost)
        return separation_loss, avg_separation_cost


class L1ReadoutLoss(nn.Module):
    def __init__(self, class_specific=True):
        super().__init__()
        self.class_specific = class_specific

    def forward(self, model: ProtoNet) -> Tensor:
        last_layer = model.last_layer
        if isinstance(last_layer, GroupedLinear):
            l1_loss = last_layer.weight.norm(p=1)
        else:
            if self.class_specific:
                l1_mask = 1 - torch.t(model.prototype_class_identity)
                l1_loss = (last_layer.weight * l1_mask).norm(p=1)
            else:
                l1_loss = last_layer.weight.norm(p=1)

        return l1_loss
