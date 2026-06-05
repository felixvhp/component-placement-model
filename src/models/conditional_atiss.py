import torch
from torch import nn

from src.config import (
    box_dim,
    height_limits,
    latent_dim,
    length_limits,
    width_limits,
)
from src.models.dit_block import BoxDiTBlock, zero_init_adaln

volume_limits = [0.005, 0.1]

D_MODEL = 256
N_HEAD = 8
N_LAYERS = 4
DIM_FF = 512
DROPOUT = 0.0
MAX_SEQ_LEN = 25

_AUX_IN = 3 + D_MODEL + 1


class AuxiliaryPredictor(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.predictor = nn.Sequential(
            nn.Linear(_AUX_IN, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, xyz, h_last, seq_len):
        feed = torch.cat([xyz, h_last, seq_len], dim=-1)
        return self.predictor(feed)


class FourierEncoding(nn.Module):
    def __init__(self, proj_dims, val=0.1):
        super().__init__()
        ll = proj_dims // 2
        exb = 2 * torch.linspace(0, ll - 1, ll) / proj_dims
        self.sigma = 2 * torch.pi / torch.pow(val, exb)

    def forward(self, x):
        sigma = self.sigma.to(x.device).view(1, -1)
        return torch.cat([torch.sin(x * sigma), torch.cos(x * sigma)], dim=-1)


class ConstraintEncoder(nn.Module):
    """
    Encodes a signed-relation target tensor [batch, N, N, 3] into a
    conditioning vector [batch, D_MODEL] that gets mixed into adaLN.
    """

    def __init__(self, max_n=MAX_SEQ_LEN, d_model=D_MODEL):
        super().__init__()
        self.max_n = max_n
        self.mlp = nn.Sequential(
            nn.Linear(max_n * max_n * 3, 256),
            nn.SiLU(),
            nn.Linear(256, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        # Zero-init final layer so encoder starts as no-op (adaLN-Zero friendly)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, target):
        batch_size, n, _, _ = target.shape
        if n < self.max_n:
            padded = torch.zeros(
                batch_size,
                self.max_n,
                self.max_n,
                3,
                device=target.device,
                dtype=target.dtype,
            )
            padded[:, :n, :n, :] = target
            target = padded
        flat = target.reshape(batch_size, -1)
        return self.mlp(flat)


# ─────────────────────────────────────────────────────────────
#  Conditional Causal Transformer Box Generator (ATISS-style)
# ─────────────────────────────────────────────────────────────
class ConditionalCTATISSBoxGenerator(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")

        self.z_proj = nn.Sequential(
            nn.Linear(latent_dim, D_MODEL),
            nn.SiLU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        self.sl_proj = nn.Linear(1, D_MODEL)

        self.constraint_encoder = ConstraintEncoder(max_n=MAX_SEQ_LEN, d_model=D_MODEL)

        self.pe_vol = FourierEncoding(proj_dims=64)
        self.vol_combine = nn.Linear(64, D_MODEL)

        self.pe_l = FourierEncoding(proj_dims=42)
        self.pe_w = FourierEncoding(proj_dims=42)
        self.pe_h = FourierEncoding(proj_dims=42)
        self.pe_x = FourierEncoding(proj_dims=42)
        self.pe_y = FourierEncoding(proj_dims=42)
        self.pe_z = FourierEncoding(proj_dims=42)
        self.box_combine = nn.Linear(252, D_MODEL)
        self.slot_combine = nn.Linear(D_MODEL + D_MODEL + D_MODEL, D_MODEL)

        self.pos_emb = nn.Embedding(MAX_SEQ_LEN + 4, D_MODEL)

        self.blocks = nn.ModuleList(
            [
                BoxDiTBlock(
                    D_MODEL, N_HEAD, mlp_ratio=DIM_FF / D_MODEL, dropout=DROPOUT
                )
                for _ in range(N_LAYERS)
            ]
        )
        zero_init_adaln(self.blocks)

        self.output_head = nn.Sequential(
            nn.Linear(D_MODEL + 1, D_MODEL),
            nn.LeakyReLU(),
            nn.Linear(D_MODEL, box_dim),
        )

        self.auxiliary_predictor = AuxiliaryPredictor(device=self.device)

    def _pos(self, idx: int, device) -> torch.Tensor:
        return self.pos_emb(torch.tensor(idx, device=device))

    @staticmethod
    def _scale_box(raw: torch.Tensor) -> torch.Tensor:
        dims = torch.sigmoid(raw[:, :3]) + 1e-4
        pos = raw[:, 3:]
        box = torch.cat([dims, pos], dim=1)
        box = box.clone()
        box[:, 0] = box[:, 0] * (length_limits[1] - length_limits[0]) + length_limits[0]
        box[:, 1] = box[:, 1] * (width_limits[1] - width_limits[0]) + width_limits[0]
        box[:, 2] = box[:, 2] * (height_limits[1] - height_limits[0]) + height_limits[0]
        box[:, 4] = 0.0
        return box

    def forward(
        self,
        volume: torch.Tensor,
        latent_z: torch.Tensor,
        target: torch.Tensor = None,
    ):
        """
        volume  : [batch, seq_len]
        latent_z: [batch, latent_dim]
        target  : [batch, seq_len, seq_len, 3] or None
        """
        assert volume.ndim == 2, "volume must be [batch, seq_len]"
        assert latent_z.ndim == 2, "latent_z must be [batch, latent_dim]"

        batch_size, seq_len = volume.shape
        device = volume.device

        boxes = torch.zeros(batch_size, seq_len, box_dim, device=device)
        z_prediction = torch.zeros(batch_size, latent_dim, device=device)

        seq_len_t = seq_len * torch.ones(batch_size, 1, device=device)

        # adaLN conditioning: latent z (+ constraint encoding if provided)
        c = self.z_proj(latent_z)
        if target is not None:
            c = c + self.constraint_encoder(target)

        sl_tok = self.sl_proj(seq_len_t) + self._pos(0, device)
        current_seq = sl_tok.unsqueeze(1)

        for i in range(seq_len):
            q_tok = self.vol_combine(self.pe_vol(volume[:, i].unsqueeze(-1)))
            seq_in = torch.cat([current_seq, q_tok.unsqueeze(1)], dim=1)

            h = seq_in
            for block in self.blocks:
                h = block(h, c)
            h_last = h[:, -1, :]

            raw = self.output_head(torch.cat([h_last, volume[:, i : i + 1]], dim=-1))
            box = self._scale_box(raw)
            boxes[:, i, :] = box

            z_add = self.auxiliary_predictor(box[:, 3:], h_last, seq_len_t)
            z_prediction = z_prediction + z_add

            box_feat = torch.cat(
                [
                    self.pe_l(box[:, 0:1]),
                    self.pe_w(box[:, 1:2]),
                    self.pe_h(box[:, 2:3]),
                    self.pe_x(box[:, 3:4]),
                    self.pe_y(box[:, 4:5]),
                    self.pe_z(box[:, 5:6]),
                ],
                dim=-1,
            )

            slot_tok = self.slot_combine(
                torch.cat(
                    [
                        self.vol_combine(self.pe_vol(volume[:, i].unsqueeze(-1))),
                        self.box_combine(box_feat),
                        self._pos(1 + i, device).expand(batch_size, -1),
                    ],
                    dim=-1,
                )
            )

            current_seq = torch.cat([current_seq, slot_tok.unsqueeze(1)], dim=1)

        return boxes, z_prediction
