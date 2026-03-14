import torch
from torch import nn

latent_dim = 4
requirement_dim = 1  # volume
box_dim = 6  # (l, w, h, x, y, z)

length_limits = [0.1, 1.5]
width_limits = [0.1, 0.5]
height_limits = [0.1, 0.5]

volume_limits = [0.005, 0.1]

D_MODEL = 128
N_HEAD = 4
N_LAYERS = 3
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
        """
        xyz     : [batch, 3]
        h_last  : [batch, D_MODEL]
        seq_len : [batch, 1]
        """
        feed = torch.cat([xyz, h_last, seq_len], dim=-1)
        return self.predictor(feed)


# ─────────────────────────────────────────────────────────────
#  Causal Transformer Box Generator
# ─────────────────────────────────────────────────────────────


class CausalTransformerBoxGenerator(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")

        self.z_proj = nn.Linear(latent_dim, D_MODEL)
        self.sl_proj = nn.Linear(1, D_MODEL)
        self.vol_proj = nn.Linear(1, D_MODEL)
        self.box_proj = nn.Linear(box_dim, D_MODEL)

        # positional embeddings to be removed eventually
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN + 4, D_MODEL)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEAD,
            dim_feedforward=DIM_FF,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True,  # pre-LN normalization more stable?
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)

        # output head maybe add latent code later
        self.output_head = nn.Sequential(
            nn.Linear(D_MODEL + 1, D_MODEL),
            # nn.Linear(D_MODEL, D_MODEL),
            nn.LeakyReLU(),
            nn.Linear(D_MODEL, box_dim),
        )

        self.auxiliary_predictor = AuxiliaryPredictor(device=self.device)

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
        z_tok = self.z_proj(latent_z) + self._pos(0, device)
        sl_tok = self.sl_proj(seq_len_t) + self._pos(1, device)

        # initial token matrix
        current_seq = torch.stack([z_tok, sl_tok], dim=1)

        for i in range(seq_len):
            # pos
            q_tok = self.vol_proj(volume[:, i].unsqueeze(-1)) + self._pos(2 + i, device)

            # query token
            seq_in = torch.cat([current_seq, q_tok.unsqueeze(1)], dim=1)

            h = self.transformer(seq_in)
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
            slot_tok = (
                self.vol_proj(volume[:, i].unsqueeze(-1))
                + self.box_proj(box)
                + self._pos(2 + i, device)
            )

            current_seq = torch.cat([current_seq, slot_tok.unsqueeze(1)], dim=1)

        return boxes, z_prediction


# ─────────────────────────────────────────────────────────────
#  Losses  (model-agnostic, identical to original baseline)
# ─────────────────────────────────────────────────────────────


def mi_loss(latent_codes, latent_code_guess, temperature=0.1):
    """InfoNCE mutual-information loss between true and predicted latent codes."""
    batch_size = latent_codes.shape[0]

    latent_codes = torch.nn.functional.normalize(latent_codes, dim=1)
    latent_code_guess = torch.nn.functional.normalize(latent_code_guess, dim=1)

    sim_matrix = torch.matmul(latent_code_guess, latent_codes.T) / temperature
    labels = torch.arange(batch_size, device=latent_codes.device)
    return torch.nn.functional.cross_entropy(sim_matrix, labels)


def DPP_loss(x, sigma=1, eps=1e-6):
    """
    Determinantal Point Process loss over the batch.
    Encourages diverse box configurations across the batch.
    x : [batch_size, seq_len, box_dim]
    """
    batch_size, seq_len, _ = x.shape
    assert seq_len > 1, "seq_len must be > 1 for DPP loss"

    x_flat = x.reshape(batch_size, -1)

    mean, std = x_flat.mean(dim=0), x_flat.std(dim=0)
    x_norm = (x_flat - mean) / (std + 1e-6)

    D2 = torch.cdist(x_norm, x_norm).pow(2)
    K = torch.exp(-D2 / (2 * sigma**2))

    return _log_det(K, batch_size, eps)


def DPP_AR_loss(x, sigma=1, eps=1e-6):
    """DPP loss on flattened box sequences."""
    x = x.view(x.size(0), -1)
    D2 = torch.cdist(x, x).pow(2)
    K = torch.exp(-D2 / (2 * sigma**2))
    return _log_det(K, K.shape[0], eps)


def DPP_loss_adjacency(A, sigma=1, eps=1e-6):
    """DPP loss on adjacency matrices. A : [batch_size, n, n]."""
    batch_size, n1, n2 = A.shape
    assert n1 == n2, "A must be square"
    A_flat = A.reshape(batch_size, -1)
    D2 = torch.cdist(A_flat, A_flat).pow(2)
    K = torch.exp(-D2 / (2 * sigma**2))
    return _log_det(K, batch_size, eps)


def _log_det(K, n, eps=1e-6):
    """Shared log-determinant computation used by all DPP losses."""
    try:
        K = K + torch.eye(n, device=K.device) * eps
        Lc = torch.linalg.cholesky(K)
        return -2 * torch.log(torch.diagonal(Lc)).sum() / n
    except Exception:
        ev = torch.linalg.eigvalsh(K.cpu())
        ev = torch.clamp(ev, min=eps)
        return -(torch.log(ev).sum() / n).to(K.device)
