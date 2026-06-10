import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

import wandb
from cpm.config import latent_dim, volume_limits
from cpm.constraints import make_constraint_fn
from cpm.geometry.internals import BatchInternals
from cpm.losses import (
    DPP_AR_loss,
    DPP_loss_adjacency,
    mi_loss,
)

# ─────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        choices=["mlp", "ctatiss", "parallel", "cond_parallel"],
        default="ctatiss",
    )
    p.add_argument("--training", choices=["ramp", "projector"], default="ramp")
    p.add_argument("--epochs", type=int, default=2500)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-seq-length", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", type=str, default="")
    p.add_argument(
        "--ablation",
        type=str,
        default="baseline",
        choices=[
            "baseline",
            "z_token",
            "z_both",
            "linear_vol",
            "no_vol_concat",
            "small",
            "large",
        ],
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────
#  Model factory
# ─────────────────────────────────────────────────────


def make_model(name, device):
    from cpm.models.atiss import CTATISSBoxGenerator
    from cpm.models.baseline import AutoRegressiveBoxGenerator
    from cpm.models.conditional_parallel import ConditionalParallelBoxGenerator
    from cpm.models.parallel_transformer import ParallelBoxGenerator

    return {
        "mlp": AutoRegressiveBoxGenerator,
        "ctatiss": CTATISSBoxGenerator,
        "parallel": ParallelBoxGenerator,
        "cond_parallel": ConditionalParallelBoxGenerator,
    }[name]().to(device)


# ─────────────────────────────────────────────────────
#  Loss computation (shared across all modes)
# ─────────────────────────────────────────────────────


def compute_losses(
    bx_pred,
    target_volumes,
    latent_codes,
    z_guess,
    seq_length,
    target_adj=None,
    compactness_kind="simple",
):
    device = bx_pred.device
    lwh, xyz = bx_pred[:, :, :3], bx_pred[:, :, 3:]
    internals = BatchInternals(xyz, lwh)
    L = {}

    # Constraints
    overlap_matrices = internals.checkOverlap()
    L["overlap"] = (
        overlap_matrices.norm(p=2, dim=1).mean(dim=1) / target_volumes.norm(p=2, dim=1)
    ).mean()
    L["volume"] = ((internals.getBoxVolumes() - target_volumes) / target_volumes).norm(
        p=2
    )

    # Compactness
    if compactness_kind == "april":
        # using old compactness function for backwards compability, can experiment with new one
        frontal_area = internals.getRoughViewArea(view="frontal")
        side_view_area = internals.getRoughViewArea(view="side")
        top_view_area = internals.getRoughViewArea(view="top")
        bbox_volume = internals.getRoughVolume() / target_volumes.sum(dim=1)
        slenderness = (
            (frontal_area / (0.5 * side_view_area + 0.5 * top_view_area)) ** 0.5
        ).mean()
        L["compactness"] = slenderness * 0.6 + 0.4 * bbox_volume.mean()
    else:
        norm = target_volumes.sum(dim=1) ** (2 / 3)
        L["compactness"] = (internals.getRoughViewArea(view="side") / norm).mean()

    # DPP
    if seq_length > 1:
        ar = torch.log(
            torch.stack(
                [lwh[:, :, 1] / lwh[:, :, 0], lwh[:, :, 2] / lwh[:, :, 0]], dim=2
            )
        )
        Ax, _, Az = internals.getAdjacency()
        L["dpp_ar"] = DPP_AR_loss(ar)
        L["ldv"] = DPP_loss_adjacency(Ax) + DPP_loss_adjacency(Az)
    else:
        L["dpp_ar"] = torch.tensor(0.0, device=device)
        L["ldv"] = torch.tensor(0.0, device=device)

    # MI
    L["mi"] = mi_loss(latent_codes, z_guess)

    # Adjacency conformity (conditional model only)
    if target_adj is not None:
        from cpm.constraints import constraint_loss

        Ax, Ay, Az = internals.getAdjacency()
        L["conform"] = constraint_loss(torch.stack([Ax, Ay, Az], dim=-1), target_adj)

    return L


# ─────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────
def make_model_with_ablation(args, device):
    """Build a Parallel model with the requested ablation."""
    if args.model != "parallel":
        raise ValueError("Ablations only supported for --model parallel")

    from cpm.models.parallel_transformer import ParallelBoxGenerator

    # Defaults
    kwargs = dict(
        d_model=256,
        n_layers=4,
        z_conditioning="adaln",
        vol_encoding="fourier",
        vol_concat_head=True,
    )

    if args.ablation == "baseline":
        pass
    elif args.ablation == "z_token":
        kwargs["z_conditioning"] = "token"
    elif args.ablation == "z_both":
        kwargs["z_conditioning"] = "both"
    elif args.ablation == "linear_vol":
        kwargs["vol_encoding"] = "linear"
    elif args.ablation == "no_vol_concat":
        kwargs["vol_concat_head"] = False
    elif args.ablation == "small":
        kwargs["d_model"] = 128
        kwargs["n_layers"] = 2
    elif args.ablation == "large":
        kwargs["d_model"] = 512
        kwargs["n_layers"] = 6
    else:
        raise ValueError(f"unknown ablation: {args.ablation}")

    return ParallelBoxGenerator(**kwargs).to(device)


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
        from cpm.config import objective_weights as ow
        from cpm.config import projector_config as pcfg
        from cpm.constraints import gated_ema_update, lm_project_step

        lambda_c = torch.zeros(2, device=device)
        lambda_d = torch.zeros(1, device=device)
        grad_clip = pcfg["grad_clip"]
    else:
        from cpm.config import get_loss_weights

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
                from cpm.constraints import sample_target

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

            compactness_kind = "april" if is_cond else "simple"
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
    if is_cond:
        suffix = f"{suffix}_april"
    name = f"{args.model}_{args.training}_plain{suffix}_s{args.seed}.pth"
    path = os.path.join(os.path.dirname(__file__), "experiments", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved: {path}")
    wandb.finish()


if __name__ == "__main__":
    train(parse_args())
