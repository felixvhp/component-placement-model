import torch
from torch import nn

from cpm.config import (
    box_dim,
    height_limits,
    latent_dim,
    length_limits,
    width_limits,
)
from cpm.models.components import AuxiliaryPredictor, FourierEncoding
from cpm.models.dit_block import BoxDiTBlock, zero_init_adaln

volume_limits = [0.005, 0.1]

D_MODEL = 256
N_HEAD = 8
N_LAYERS = 4
DIM_FF = 512
DROPOUT = 0.0
MAX_SEQ_LEN = 25


class CTATISSBoxGenerator(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")

        self.z_proj = nn.Sequential(
            nn.Linear(latent_dim, D_MODEL),
            nn.SiLU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        self.sl_proj = nn.Linear(1, D_MODEL)

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

        # positional embeddings to be removed eventually
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

        # output head maybe add latent code later
        self.output_head = nn.Sequential(
            nn.Linear(D_MODEL + 1, D_MODEL),
            # nn.Linear(D_MODEL, D_MODEL),
            nn.LeakyReLU(),
            nn.Linear(D_MODEL, box_dim),
        )

        self.auxiliary_predictor = AuxiliaryPredictor(D_MODEL, device=self.device)

    def _pos(self, idx: int, device) -> torch.Tensor:
        """Return the positional embedding vector for position `idx`."""
        return self.pos_emb(torch.tensor(idx, device=device))

    @staticmethod
    def _scale_box(raw: torch.Tensor) -> torch.Tensor:
        """
        Apply activations and physical scaling
        """
        # Sigmoid constrains dimensions to (0, 1), then scale to physical range
        dims = torch.sigmoid(raw[:, :3]) + 1e-4
        pos = raw[:, 3:]  # positions are unconstrained

        box = torch.cat([dims, pos], dim=1)

        # Scale each dimension into its physical limits
        box = box.clone()
        box[:, 0] = box[:, 0] * (length_limits[1] - length_limits[0]) + length_limits[0]
        box[:, 1] = box[:, 1] * (width_limits[1] - width_limits[0]) + width_limits[0]
        box[:, 2] = box[:, 2] * (height_limits[1] - height_limits[0]) + height_limits[0]

        box[:, 4] = 0.0

        return box

    def forward(self, volume: torch.Tensor, latent_z: torch.Tensor):
        assert volume.ndim == 2, "volume must be [batch, seq_len]"
        assert latent_z.ndim == 2, "latent_z must be [batch, latent_dim]"

        batch_size, seq_len = volume.shape
        device = volume.device

        boxes = torch.zeros(batch_size, seq_len, box_dim, device=device)
        z_prediction = torch.zeros(batch_size, latent_dim, device=device)

        seq_len_t = seq_len * torch.ones(batch_size, 1, device=device)

        # context tokens - positional embeddings to be removed
        c = self.z_proj(latent_z)  # [batch, D_MODEL] — adaLN conditioning
        sl_tok = self.sl_proj(seq_len_t) + self._pos(0, device)  # shifted to pos 0
        current_seq = sl_tok.unsqueeze(1)

        for i in range(seq_len):
            # qury token
            q_tok = self.vol_combine(self.pe_vol(volume[:, i].unsqueeze(-1)))

            # add it
            seq_in = torch.cat([current_seq, q_tok.unsqueeze(1)], dim=1)

            h = seq_in
            for block in self.blocks:
                h = block(h, c)
            h_last = h[:, -1, :]

            # predict box
            # raw = self.output_head(h_last)  # [batch, box_dim]
            raw = self.output_head(torch.cat([h_last, volume[:, i : i + 1]], dim=-1))
            box = self._scale_box(raw)
            boxes[:, i, :] = box

            # auxiliary predict
            z_add = self.auxiliary_predictor(box[:, 3:], h_last, seq_len_t)
            z_prediction = z_prediction + z_add

            # new box token
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
