import torch
from torch import nn

from src.config import (
    box_dim,
    height_limits,
    latent_dim,
    length_limits,
    width_limits,
)
from src.models.dit_block import BoxDiTBlock, BoxDiTCrossAttnBlock, zero_init_adaln
from src.models.components import AuxiliaryPredictor, FourierEncoding

# Defaults (used when no args passed)
DEFAULT_D_MODEL = 256
DEFAULT_N_HEAD = 8
DEFAULT_N_LAYERS = 4
DEFAULT_DIM_FF = 512
DEFAULT_DROPOUT = 0.0
MAX_SEQ_LEN = 25


class ParallelBoxGenerator(nn.Module):
    def __init__(
        self,
        device=None,
        d_model=DEFAULT_D_MODEL,
        n_heads=DEFAULT_N_HEAD,
        n_layers=DEFAULT_N_LAYERS,
        dim_ff=None,
        dropout=DEFAULT_DROPOUT,
        # Ablation switches
        z_conditioning="adaln",      # "adaln" | "token | both"
        vol_encoding="fourier",      # "fourier" | "linear"
        vol_concat_head=True,        # True | False
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.d_model = d_model
        self.n_layers = n_layers
        self.dim_ff = dim_ff if dim_ff is not None else 2 * d_model
        self.z_conditioning = z_conditioning
        self.vol_encoding = vol_encoding
        self.vol_concat_head = vol_concat_head

        # z projection
        self.z_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # seq_len context token
        self.sl_proj = nn.Linear(1, d_model)

        # Volume encoding
        if vol_encoding == "fourier":
            self.pe_vol = FourierEncoding(proj_dims=64)
            self.vol_proj = nn.Linear(64, d_model)
        elif vol_encoding == "linear":
            self.pe_vol = None
            self.vol_proj = nn.Linear(1, d_model)
        else:
            raise ValueError(f"unknown vol_encoding: {vol_encoding}")

        # Transformer blocks
        self.self_attn_blocks = nn.ModuleList([
            BoxDiTBlock(d_model, n_heads, mlp_ratio=self.dim_ff / d_model, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.cross_attn_blocks = nn.ModuleList([
            BoxDiTCrossAttnBlock(d_model, n_heads, mlp_ratio=self.dim_ff / d_model, dropout=dropout)
            for _ in range(n_layers)
        ])
        zero_init_adaln(self.self_attn_blocks)
        zero_init_adaln(self.cross_attn_blocks)

        # Output head
        head_in = d_model + 1 if vol_concat_head else d_model
        self.output_head = nn.Sequential(
            nn.Linear(head_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, box_dim),
        )

        self.query_embed = nn.Embedding(MAX_SEQ_LEN, d_model)

        self.auxiliary_predictor = AuxiliaryPredictor(d_model, device=self.device)

    @staticmethod
    def _scale_box(raw: torch.Tensor) -> torch.Tensor:
        dims = torch.sigmoid(raw[..., :3]) + 1e-4
        pos = raw[..., 3:]
        box = torch.cat([dims, pos], dim=-1).clone()
        box[..., 0] = box[..., 0] * (length_limits[1] - length_limits[0]) + length_limits[0]
        box[..., 1] = box[..., 1] * (width_limits[1] - width_limits[0]) + width_limits[0]
        box[..., 2] = box[..., 2] * (height_limits[1] - height_limits[0]) + height_limits[0]
        box[..., 4] = 0.0
        return box

    def forward(self, volume, latent_z):
        assert volume.ndim == 2
        assert latent_z.ndim == 2
        batch_size, seq_len = volume.shape
        device = volume.device
        seq_len_t = seq_len * torch.ones(batch_size, 1, device=device)

        # ── z conditioning ──────────────────────────────
        z_emb = self.z_proj(latent_z)  # [B, d_model]

        if self.z_conditioning == "adaln":
            c = z_emb
            context_tokens = []
        elif self.z_conditioning == "token":
            c = torch.zeros(batch_size, self.d_model, device=device)
            context_tokens = [z_emb.unsqueeze(1)]
        elif self.z_conditioning == "both":
            c = z_emb
            context_tokens = [z_emb.unsqueeze(1)]
        else:
            raise ValueError(f"unknown z_conditioning: {self.z_conditioning}")

        # seq_len token always in context
        sl_token = self.sl_proj(seq_len_t).unsqueeze(1)
        context_tokens.append(sl_token)
        context = torch.cat(context_tokens, dim=1)

        # ── volume encoding ─────────────────────────────
        if self.vol_encoding == "fourier":
            vol_features = self.pe_vol(volume.unsqueeze(-1))
            q = self.vol_proj(vol_features)
        else:  # linear
            q = self.vol_proj(volume.unsqueeze(-1))

        slot_ids = torch.arange(seq_len, device=device)
        q = q + self.query_embed(slot_ids).unsqueeze(0)

        # ── transformer blocks ──────────────────────────
        for self_block, cross_block in zip(self.self_attn_blocks, self.cross_attn_blocks):
            q = self_block(q, c)
            q = cross_block(q, context, c)

        # ── output head ─────────────────────────────────
        if self.vol_concat_head:
            head_in = torch.cat([q, volume.unsqueeze(-1)], dim=-1)
        else:
            head_in = q
        raw = self.output_head(head_in)
        boxes = self._scale_box(raw)

        # ── aux z predictor ─────────────────────────────
        z_prediction = torch.zeros(batch_size, latent_dim, device=device)
        for i in range(seq_len):
            z_add = self.auxiliary_predictor(boxes[:, i, 3:], q[:, i, :], seq_len_t)
            z_prediction = z_prediction + z_add

        return boxes, z_prediction
