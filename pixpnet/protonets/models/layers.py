# Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
from torch import nn


class GroupedLinear(nn.Linear):
    """
    Equivalent to (but faster than):
    >>> conv = nn.Conv1d(in_features, out_features, kernel_size=1,
    >>>                  groups=groups, bias=bias, **kwargs)
    >>> conv(input[:, :, None]).squeeze(dim=2)
    """

    def __init__(self, in_features, out_features, groups, bias=True, **kwargs):
        if in_features % groups != 0:
            raise ValueError("in_features must be divisible by groups")

        self.groups = groups
        self.in_features_per_group = in_features // groups

        super().__init__(in_features=self.in_features_per_group, out_features=out_features, bias=bias, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.groups == 1:
            return super().forward(input)
        # Otherwise, group input
        input_grouped = torch.reshape(input, (input.size()[0], self.groups, self.in_features_per_group))
        # Batched matrix multiplications using einsum
        out = torch.einsum("bji,ji->bj", input_grouped, self.weight)
        # Add bias if using bias
        if self.bias is not None:
            out += self.bias[None, :]
        return out


class GroupedSum(nn.Module):
    def __init__(self, in_features, groups):
        if in_features % groups != 0:
            raise ValueError("in_features must be divisible by groups")

        self.groups = groups
        self.in_features_per_group = in_features // groups
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.groups == 1:
            return input
        # Otherwise, group input
        input_grouped = torch.reshape(input, (input.size()[0], self.groups, self.in_features_per_group))
        return torch.sum(input_grouped, dim=2)
