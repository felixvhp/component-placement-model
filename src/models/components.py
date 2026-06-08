"""Shared building blocks used across the box-generator model variants."""

import torch
from torch import nn


class FourierEncoding(nn.Module):
    def __init__(self, proj_dims, val=0.1):
        super().__init__()
        ll = proj_dims // 2
        exb = 2 * torch.linspace(0, ll - 1, ll) / proj_dims
        self.sigma = 2 * torch.pi / torch.pow(val, exb)

    def forward(self, x):
        sigma = self.sigma.to(x.device).view(1, -1)
        return torch.cat([torch.sin(x * sigma), torch.cos(x * sigma)], dim=-1)
