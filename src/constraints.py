"""
Constraint utilities for conditional box generation.

Targets are signed per-axis relations matching the shape produced by
BatchInternals.getAdjacency() — specifically, a tensor of shape
[batch, N, N, 3] where the last axis is (x, y, z) and each entry is in
{-target_strength, 0, +target_strength}:

    +s  : box i is on the +axis side of box j
    -s  : box i is on the -axis side of box j
     0  : no constraint

Entries are antisymmetric in (i, j) and the diagonal is zero.
"""

import torch


def sample_target(
    batch_size,
    seq_len,
    density=0.3,
    target_strength=0.3,
    axes=(0, 2),  # skip y by default (y=0 symmetry enforced by the model)
    device="cpu",
):
    """
    Sample a sparse signed-relation target.

    For each example, each upper-triangular pair (i, j) with i < j is
    independently selected with probability `density`. Selected pairs get a
    random axis from `axes` and a random sign. The antisymmetric counterpart
    in the lower triangle is set automatically.

    `density` may be a Python scalar or a 1-D tensor of shape [batch_size]
    (per-example density). Per-example values are clamped to [0, 1].

    Returns: [batch_size, seq_len, seq_len, 3] tensor on `device`.
    """
    target = torch.zeros(batch_size, seq_len, seq_len, 3, device=device)

    if seq_len < 2:
        return target  # no pairs to constrain

    # Upper-triangular index pairs (i < j)
    i_idx, j_idx = torch.triu_indices(seq_len, seq_len, offset=1, device=device)
    n_pairs = i_idx.numel()

    # Per-example, per-pair selection mask. `density` is a scalar or [B] tensor.
    if torch.is_tensor(density):
        density = density.to(device).clamp(0.0, 1.0).view(batch_size, 1)
    active = torch.rand(batch_size, n_pairs, device=device) < density  # [B, P]

    # Random axis choice (index into `axes`) and random sign per pair
    axes_t = torch.tensor(axes, device=device)
    axis_choice = axes_t[torch.randint(0, len(axes), (batch_size, n_pairs), device=device)]
    sign = torch.where(
        torch.rand(batch_size, n_pairs, device=device) < 0.5,
        torch.full((batch_size, n_pairs), target_strength, device=device),
        torch.full((batch_size, n_pairs), -target_strength, device=device),
    )

    # Scatter into the target tensor
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, n_pairs)
    # Flatten to index assignment
    b_flat = batch_idx[active]
    i_flat = i_idx.unsqueeze(0).expand(batch_size, -1)[active]
    j_flat = j_idx.unsqueeze(0).expand(batch_size, -1)[active]
    axis_flat = axis_choice[active]
    sign_flat = sign[active]

    target[b_flat, i_flat, j_flat, axis_flat] = sign_flat
    target[b_flat, j_flat, i_flat, axis_flat] = -sign_flat  # antisymmetric

    return target


def constraint_loss(measured, target, eps=1e-6):
    """
    Masked MSE between the model's measured signed ordering and the target.

    measured : [batch, N, N, 3] from torch.stack([Ax, Ay, Az], dim=-1)
    target   : [batch, N, N, 3] from sample_target

    Only entries where target != 0 contribute to the loss. Returns a scalar.
    """
    pair_mask = (target != 0).any(dim=-1, keepdim=True).float()  # [batch, N, N, 1]
    # Broadcast to all 3 axes
    full_mask = pair_mask.expand_as(target)                       # [batch, N, N, 3]

    total = full_mask.sum()
    if total.item() == 0:
        return torch.zeros((), device=measured.device, dtype=measured.dtype)

    return ((measured - target) ** 2 * full_mask).sum() / (total + eps)


def satisfaction_rate(measured, target):
    """
    Fraction of constrained entries whose measured sign matches the target sign.

    Useful as an eval metric (not a training loss — uses a hard sign comparison).
    Returns a scalar in [0, 1]. If no constraints are specified, returns 1.0.
    """
    mask = target != 0
    if mask.sum().item() == 0:
        return torch.ones((), device=measured.device, dtype=measured.dtype)
    correct = (torch.sign(measured[mask]) == torch.sign(target[mask])).float()
    return correct.mean()
