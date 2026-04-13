import torch
from torch import nn


def modulate(x, shift, scale):
    """Apply adaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BoxDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        # LayerNorms without learnable affine — adaLN provides scale/shift
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Self-attention
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        # MLP
        mlp_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_size),
        )

        # adaLN modulation: c -> (shift_attn, scale_attn, gate_attn,
        #                          shift_mlp,  scale_mlp,  gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        """
        x: [batch, seq_len, hidden_size] — token sequence
        c: [batch, hidden_size]           — conditioning vector (projected z)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        # Self-attention with adaLN
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP with adaLN
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x


class BoxDiTCrossAttnBlock(nn.Module):
    """
    adaLN-Zero block with cross-attention.
    Query cross-attends to context, both modulated by z.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        self.norm_q = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_kv = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        mlp_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_size),
        )

        # 6 params: shift/scale/gate for cross-attn + shift/scale/gate for MLP
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, q, kv, c):
        """
        q:  [batch, 1, hidden_size]     — query token (volume)
        kv: [batch, seq_len, hidden_size] — context (sl_tok + previous boxes)
        c:  [batch, hidden_size]          — conditioning (projected z)
        """
        shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        # Cross-attention with adaLN
        q_norm = modulate(self.norm_q(q), shift_ca, scale_ca)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm)
        q = q + gate_ca.unsqueeze(1) * attn_out

        # MLP with adaLN
        q = q + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(q), shift_mlp, scale_mlp)
        )

        return q


class FinalAdaLN(nn.Module):
    """
    Final layer with adaLN modulation before the output projection.
    Applies shift/scale from c to the final LayerNorm, then projects to output_dim.
    """

    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        """
        x: [batch, hidden_size] — single token representation
        c: [batch, hidden_size] — conditioning vector
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = x * (1 + scale) + shift  # no unsqueeze needed, x is 2D here
        x = self.norm(x)
        x = self.linear(x)
        return x


def zero_init_adaln(blocks, final_layer=None):
    """
    Zero-initialize all adaLN modulation layers so each block starts as identity.
    Call this after model construction.
    """
    for block in blocks:
        nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    if final_layer is not None:
        nn.init.constant_(final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(final_layer.adaLN_modulation[-1].bias, 0)
