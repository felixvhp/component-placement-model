"""The training loop: builds the model, runs the ramp or projector objective,
logs to W&B, and saves the checkpoint to ``experiments/`` (relative to cwd)."""

import os

import numpy as np
import torch
from tqdm import tqdm

import wandb
from cpm.config import (
    get_loss_weights,
    latent_dim,
    volume_limits,
)
from cpm.config import (
    objective_weights as ow,
)
from cpm.config import (
    projector_config as pcfg,
)
from cpm.constraints import (
    gated_ema_update,
    lm_project_step,
    make_constraint_fn,
    sample_target,
)
from cpm.training.factory import make_model, make_model_with_ablation
from cpm.training.objective import compute_losses


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    if args.model == "parallel":
        model = make_model_with_ablation(args, device)
    else:
        model = make_model(args.model, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    is_ar = args.model in ("mlp", "ctatiss")
    is_cond = args.model in ("cond_parallel",)
    use_proj = args.training == "projector"

    wandb.init(project="box-arrangement", name=args.run_name or None, config=vars(args))

    # Mode-specific setup
    if use_proj:
        lambda_c = torch.zeros(2, device=device)
        lambda_d = torch.zeros(1, device=device)
        grad_clip = pcfg["grad_clip"]
    else:
        grad_clip = 1.0

    for epoch in tqdm(range(args.epochs), ncols=100):
        optimizer.zero_grad()
        z = torch.randn(args.batch_size, latent_dim, device=device)

        loop_all_sl = is_ar or is_cond
        seq_lengths = (
            range(1, args.max_seq_length + 1)
            if loop_all_sl
            else [torch.randint(1, args.max_seq_length + 1, (1,)).item()]
        )

        total_obj = torch.tensor(0.0, device=device)
        total_con = torch.tensor(0.0, device=device)
        total_disp = torch.tensor(0.0, device=device)
        vol_posts, ovl_posts, disp_posts = [], [], []

        for sl in seq_lengths:
            vols = volume_limits[0] + torch.rand(args.batch_size, sl, device=device) * (
                volume_limits[1] - volume_limits[0]
            )

            # Forward
            tgt_adj = None
            if is_cond:
                tgt_adj = sample_target(
                    args.batch_size,
                    sl,
                    density=0.3,
                    target_strength=0.8,
                    device=device,
                )
                bx, zg = model(vols, z, target=tgt_adj)
            else:
                bx, zg = model(vols, z)

            compactness_kind = "slender" if is_cond else "simple"
            L = compute_losses(
                bx,
                vols,
                z,
                zg,
                sl,
                tgt_adj,
                compactness_kind=compactness_kind,
            )

            # ── Projector path ────────────────────────────
            if use_proj:
                try:
                    y_hat = bx.view(args.batch_size, -1)
                    _, c_pre, c_post, disp = lm_project_step(
                        y_hat,
                        make_constraint_fn(vols, sl),
                        delta=pcfg["delta"],
                        lam_min=pcfg["lam_min"],
                    )

                    cr = c_pre.abs()
                    total_con = (
                        total_con + (lambda_c[0].detach() * cr[:, :sl]).sum(-1).mean()
                    )
                    if sl > 1:
                        total_con = (
                            total_con
                            + (lambda_c[1].detach() * cr[:, sl:]).sum(-1).mean()
                        )

                    dm = disp.pow(2).sum(-1).add(1e-12).sqrt()
                    total_disp = total_disp + (lambda_d.detach() * dm).mean()

                    with torch.no_grad():
                        cp = c_post.abs()
                        vol_posts.append(cp[:, :sl].mean())
                        if sl > 1:
                            ovl_posts.append(cp[:, sl:].mean())
                        disp_posts.append(dm.detach().mean())
                except Exception as e:
                    tqdm.write(f"Projector failed sl={sl}: {e}")

                total_obj = total_obj + (
                    ow["compactness"] * L["compactness"]
                    + ow["dpp_aspect_ratios"] * L["dpp_ar"]
                    + ow["ldv"] * L["ldv"]
                    + ow["infonce"] * L["mi"]
                    + ow.get("conform", 30.0) * L.get("conform", 0.0)
                )

            # ── Ramp path ─────────────────────────────────
            else:
                wts = get_loss_weights(epoch)
                λo, λv, λc, λl, λa, _, λm = wts[:7]
                λcf = wts[7] if len(wts) > 7 else 0.0
                total_obj = total_obj + (
                    λo * L["overlap"]
                    + λv * L["volume"]
                    + λc * L["compactness"]
                    + λl * L["ldv"]
                    + λa * L["dpp_ar"]
                    + λm * L["mi"]
                    + λcf * L.get("conform", 0.0)
                )

        # ── Step ──────────────────────────────────────────
        total_loss = total_obj + total_con + total_disp
        if torch.isnan(total_loss):
            tqdm.write(f"NaN at epoch {epoch}")
            break

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        # ── Projector EMA ─────────────────────────────────
        if use_proj and vol_posts:
            with torch.no_grad():
                lambda_c[0] = gated_ema_update(
                    lambda_c[0],
                    torch.stack(vol_posts).mean(),
                    pcfg["feas_tol"],
                    pcfg["ema_max_decades"],
                    pcfg["ema_rate"],
                )
                if ovl_posts:
                    lambda_c[1] = gated_ema_update(
                        lambda_c[1],
                        torch.stack(ovl_posts).mean(),
                        pcfg["feas_tol"],
                        pcfg["ema_max_decades"],
                        pcfg["ema_rate"],
                    )
                if disp_posts:
                    lambda_d[0] = gated_ema_update(
                        lambda_d[0],
                        torch.stack(disp_posts).mean(),
                        pcfg["disp_tol"],
                        pcfg["ema_max_decades"],
                        pcfg["ema_rate"],
                    )

        # ── Log ───────────────────────────────────────────
        ld = {"total_loss": total_loss.item()}
        if use_proj:
            ld.update(
                {
                    "con_loss": total_con.item(),
                    "disp_loss": total_disp.item(),
                    "lam_c_vol": lambda_c[0].item(),
                    "lam_c_ovl": lambda_c[1].item(),
                    "lam_d": lambda_d[0].item(),
                }
            )
        wandb.log(ld, step=epoch)

        if (epoch + 1) % 100 == 0:
            tqdm.write(f"[{epoch + 1}] loss={total_loss.item():.3f}")

    # ── Save ──────────────────────────────────────────────
    suffix = f"_{args.ablation}" if args.ablation != "baseline" else ""
    name = f"{args.model}_{args.training}_plain{suffix}_s{args.seed}.pth"
    path = os.path.join("experiments", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved: {path}")
    wandb.finish()
