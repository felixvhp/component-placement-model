"""
Levenberg-Marquardt projection step with Yamashita-Fukushima damping.

Cleaned up from PAL projector (Phuc Vu / Tim Aebersold).

The projector takes the model's raw output y and a constraint function,
then computes a single Newton-like step toward the feasible manifold.
It returns both detached and live (gradient-carrying) quantities so the
training loop can build a loss that teaches the model to stay near feasible.
"""

import torch


def lm_project_step(
    y, constraint_fn, *, delta=1e-3, lam_min=1e-6, prescale=False, prescale_floor=1.0
):
    """
    One LM step projecting y toward constraint_fn(y) = 0.

    Args:
        y:              [B, D]  model output (live — gradients flow to model params)
        constraint_fn:  callable(y) -> [B, K]  constraint residuals (0 = satisfied)
        delta:          YF damping scale: lam = delta * ||c||^2
        lam_min:        minimum damping (prevents singular systems)
        prescale:       if True, normalize each Jacobian row to unit norm
        prescale_floor: minimum row norm when prescaling

    Returns:
        y_new:        [B, D]  projected point (detached)
        c_pre:        [B, K]  constraints at y (live — for constraint loss)
        c_post_live:  [B, K]  constraints at projected point (live — for EMA tracking)
        disp_live:    [B, D]  displacement y - y_tilde (live — for displacement loss)
    """
    B, D = y.shape

    # ── 1. Jacobian via autograd on detached copy ─────────────────
    #    No second-order gradients reach model params through here.
    y_det = y.detach().requires_grad_(True)
    c_det = constraint_fn(y_det)
    K = c_det.shape[-1]

    J = torch.zeros(B, K, D, device=y.device)
    for k in range(K):
        J[:, k, :] = torch.autograd.grad(
            c_det[:, k].sum(), y_det, retain_graph=(k < K - 1)
        )[0]
    J = J.detach()

    # ── 2. Live constraints at y (gradients flow to model) ────────
    c_pre = constraint_fn(y)
    c_pre_det = c_pre.detach()

    # ── 3. Optional row prescaling (unit-norm Jacobian rows) ──────
    B_mat = J
    c_for_solve = c_pre_det
    row_norms = None
    if prescale:
        row_norms = B_mat.norm(dim=2, keepdim=True).clamp(min=prescale_floor)
        B_mat = B_mat / row_norms
        c_for_solve = c_for_solve / row_norms.squeeze(-1)

    # ── 4. Yamashita-Fukushima damping ────────────────────────────
    BBT = torch.bmm(B_mat, B_mat.transpose(1, 2))  # [B, K, K]
    c_norm_sq = c_for_solve.pow(2).sum(dim=1, keepdim=True)  # [B, 1]
    lam = (delta * c_norm_sq).clamp(min=lam_min)  # [B, 1]
    eye = torch.eye(K, dtype=BBT.dtype, device=BBT.device)
    BBT_reg = BBT + lam.unsqueeze(-1) * eye  # [B, K, K]

    # ── 5. Detached solve → y_new ─────────────────────────────────
    L = torch.linalg.cholesky(BBT_reg)
    z = torch.cholesky_solve(c_for_solve.unsqueeze(-1), L).squeeze(-1)  # [B, K]
    dy = torch.bmm(B_mat.transpose(1, 2), z.unsqueeze(-1)).squeeze(-1)  # [B, D]
    y_new = y.detach() - dy

    # ── 6. Live solve → displacement with grads through y → c ─────
    c_live = c_pre
    if prescale:
        c_live = c_live / row_norms.squeeze(-1).clamp(min=prescale_floor)
    z_live = torch.cholesky_solve(c_live.unsqueeze(-1), L).squeeze(-1)
    disp_live = torch.bmm(B_mat.transpose(1, 2), z_live.unsqueeze(-1)).squeeze(-1)
    y_tilde_live = y - disp_live
    c_post_live = constraint_fn(y_tilde_live)

    return y_new, c_pre, c_post_live, disp_live


def gated_ema_update(lam, metric_mean, tol, max_decades=1.0, rate=1e-2):
    """
    Gated EMA multiplier update.

    If the metric (e.g. mean post-projection residual) is above tolerance,
    the multiplier grows. If below, it can shrink slightly. Clamped to >= 0.

    Args:
        lam:          current multiplier (scalar tensor)
        metric_mean:  batch-mean of the quantity being tracked (scalar tensor)
        tol:          target tolerance
        max_decades:  max log10 gap per step (prevents explosive growth)
        rate:         EMA learning rate

    Returns:
        updated multiplier (scalar tensor)
    """
    gap = torch.log10(metric_mean.clamp(min=1e-30) / tol).clamp(max=max_decades)
    return (lam + rate * gap).clamp(min=0.0)


# ─────────────────────────────────────────────────────
#  Constraint function for projector
# ─────────────────────────────────────────────────────


def make_constraint_fn(target_volumes, seq_len):
    def cfn(y_flat):
        B = y_flat.shape[0]
        boxes = y_flat.view(B, seq_len, 6)
        lwh, xyz = boxes[:, :, :3], boxes[:, :, 3:]

        vol_residual = (lwh.prod(dim=-1) - target_volumes) / target_volumes
        constraints = [vol_residual]

        if seq_len > 1:
            half = lwh / 2
            ci, cj = xyz.unsqueeze(2), xyz.unsqueeze(1)
            hi, hj = half.unsqueeze(2), half.unsqueeze(1)
            ovl = torch.relu(hi + hj - (ci - cj).abs()).prod(dim=-1)
            ovl = ovl * (1 - torch.eye(seq_len, device=y_flat.device).unsqueeze(0))
            constraints.append(
                (ovl.sum(dim=[1, 2]) / target_volumes.sum(dim=1)).unsqueeze(-1)
            )

        return torch.cat(constraints, dim=-1)

    return cfn
