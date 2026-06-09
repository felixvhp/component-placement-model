"""Shared building blocks used across the box-generator model variants."""

import torch
from torch import nn

from src.config import latent_dim


class FourierEncoding(nn.Module):
    def __init__(self, proj_dims, val=0.1):
        super().__init__()
        ll = proj_dims // 2
        exb = 2 * torch.linspace(0, ll - 1, ll) / proj_dims
        self.sigma = 2 * torch.pi / torch.pow(val, exb)

    def forward(self, x):
        sigma = self.sigma.to(x.device).view(1, -1)
        return torch.cat([torch.sin(x * sigma), torch.cos(x * sigma)], dim=-1)


class AuxiliaryPredictor(nn.Module):
    """Predicts a latent-code increment from a box's position, hidden state, and
    sequence length. Input dim is ``3 + d_model + 1`` (xyz, hidden, seq_len)."""

    def __init__(self, d_model, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        aux_in = 3 + d_model + 1
        self.predictor = nn.Sequential(
            nn.Linear(aux_in, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, xyz, h, seq_len):
        return self.predictor(torch.cat([xyz, h, seq_len], dim=-1))
