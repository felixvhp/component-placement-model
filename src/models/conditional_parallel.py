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

D_MODEL = 256
N_HEAD = 8
N_LAYERS = 4
DIM_FF = 512
DROPOUT = 0.0
MAX_SEQ_LEN = 25


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
        # Zero-init final layer so encoder starts as no-op (plays nice with adaLN-Zero)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, target):
        batch_size, n, _, _ = target.shape
        if n < self.max_n:
            padded = torch.zeros(
                batch_size, self.max_n, self.max_n, 3,
                device=target.device, dtype=target.dtype,
            )
            padded[:, :n, :n, :] = target
            target = padded
        flat = target.reshape(batch_size, -1)
        return self.mlp(flat)


# ─────────────────────────────────────────────────────────────
#  Conditional Parallel Box Generator (DETR-style decoder)
# ─────────────────────────────────────────────────────────────


class ConditionalParallelBoxGenerator(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")

        # Project latent z into adaLN conditioning vector
        self.z_proj = nn.Sequential(
            nn.Linear(latent_dim, D_MODEL),
            nn.SiLU(),
            nn.Linear(D_MODEL, D_MODEL),
        )

        self.constraint_encoder = ConstraintEncoder(max_n=MAX_SEQ_LEN, d_model=D_MODEL)

        # context and query tokens
        self.sl_proj = nn.Linear(1, D_MODEL)  # seq_len token
        self.pe_vol = FourierEncoding(proj_dims=64)
        self.vol_proj = nn.Linear(64, D_MODEL)

        # self-attention among queries, then cross-attention to context
        self.self_attn_blocks = nn.ModuleList(
            [
                BoxDiTBlock(
                    D_MODEL, N_HEAD, mlp_ratio=DIM_FF / D_MODEL, dropout=DROPOUT
                )
                for _ in range(N_LAYERS)
            ]
        )
        self.cross_attn_blocks = nn.ModuleList(
            [
                BoxDiTCrossAttnBlock(
                    D_MODEL, N_HEAD, mlp_ratio=DIM_FF / D_MODEL, dropout=DROPOUT
                )
                for _ in range(N_LAYERS)
            ]
        )

        zero_init_adaln(self.self_attn_blocks)
        zero_init_adaln(self.cross_attn_blocks)

        # Ccncatenate volume scalar to hidden state before prediction
        self.output_head = nn.Sequential(
            nn.Linear(D_MODEL + 1, D_MODEL),
            nn.LeakyReLU(),
            nn.Linear(D_MODEL, box_dim),
        )

        self.query_embed = nn.Embedding(MAX_SEQ_LEN, D_MODEL)

        # ── Auxiliary z predictor (for MI loss) ───────────────
        self.auxiliary_predictor = AuxiliaryPredictor(D_MODEL, device=self.device)

    @staticmethod
    def _scale_box(raw: torch.Tensor) -> torch.Tensor:
        """
        Apply activations and physical scaling.
        Works on [batch, seq_len, box_dim] — batched over both dimensions.
        """
        dims = torch.sigmoid(raw[..., :3]) + 1e-4  # constrain to (0, 1)
        pos = raw[..., 3:]  # positions unconstrained

        box = torch.cat([dims, pos], dim=-1)
        box = box.clone()

        # Scale dimensions into physical limits
        box[..., 0] = (
            box[..., 0] * (length_limits[1] - length_limits[0]) + length_limits[0]
        )
        box[..., 1] = (
            box[..., 1] * (width_limits[1] - width_limits[0]) + width_limits[0]
        )
        box[..., 2] = (
            box[..., 2] * (height_limits[1] - height_limits[0]) + height_limits[0]
        )

        # Enforce y=0 symmetry
        box[..., 4] = 0.0

        return box

    def forward(self, volume: torch.Tensor, latent_z: torch.Tensor, target: torch.Tensor = None):
        """
        volume  : [batch, seq_len]       — target volumes for each box
        latent_z: [batch, latent_dim]    — latent conditioning code
        target  : [batch, seq_len, seq_len, 3] or None — signed-relation constraints

        Returns:
            boxes       : [batch, seq_len, box_dim]
            z_prediction: [batch, latent_dim]
        """
        assert volume.ndim == 2, "volume must be [batch, seq_len]"
        assert latent_z.ndim == 2, "latent_z must be [batch, latent_dim]"

        batch_size, seq_len = volume.shape
        device = volume.device

        seq_len_t = seq_len * torch.ones(batch_size, 1, device=device)

        # conditioning for adaln
        c = self.z_proj(latent_z)

        # NEW: add constraint encoding into adaLN conditioning
        if target is not None:
            c = c + self.constraint_encoder(target)

        # cross attention conyrxt
        context = self.sl_proj(seq_len_t).unsqueeze(1)
        z_token = c.unsqueeze(1)
        sl_token = self.sl_proj(seq_len_t).unsqueeze(1)
        context = torch.cat([z_token, sl_token], dim=1)

        # make query vectors out volumes
        vol_features = self.pe_vol(volume.unsqueeze(-1))
        slot_ids = torch.arange(seq_len, device=device)
        q = self.vol_proj(vol_features) + self.query_embed(slot_ids).unsqueeze(0)

        # self attention cross attention blocks
        for self_block, cross_block in zip(
            self.self_attn_blocks, self.cross_attn_blocks
        ):
            q = self_block(q, c)
            q = cross_block(q, context, c)

        raw = self.output_head(
            torch.cat([q, volume.unsqueeze(-1)], dim=-1)
        )  # [batch, seq_len, box_dim]

        boxes = self._scale_box(raw)

        z_prediction = torch.zeros(batch_size, latent_dim, device=device)
        for i in range(seq_len):
            z_add = self.auxiliary_predictor(
                boxes[:, i, 3:],
                q[:, i, :],
                seq_len_t,
            )
            z_prediction = z_prediction + z_add

        return boxes, z_prediction
