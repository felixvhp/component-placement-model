import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
import wandb

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.geometry.internals import BatchInternals
from src.losses import (
    DPP_AR_loss, DPP_loss_adjacency,
    PA_DPP_AR_loss, PA_DPP_loss_adjacency,
    mi_loss,
)
from src.config import volume_limits, latent_dim


# ─────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mlp", "ctatiss", "parallel", "cond_parallel"], default="ctatiss")
    p.add_argument("--training", choices=["ramp", "projector"], default="ramp")
    p.add_argument("--dpp", choices=["plain", "padpp"], default="plain")
    p.add_argument("--epochs", type=int, default=2500)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-seq-length", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--ablation", type=str, default="baseline",
               choices=["baseline", "z_token", "z_both", "linear_vol", "no_vol_concat",
                        "small", "large", "no_mi"])
    p.add_argument("--cond-recipe", type=str, default="default",
               choices=["default", "april_unif", "april"],
               help="Conditional training recipe. "
                    "'default' = single random sl/epoch, density=0.3, target_strength=0.3, "
                    "simple side-area compactness. "
                    "'april_unif' = loop sl=1..max_sl/epoch, per-example density U(0, 0.5), "
                    "target_strength=0.8, simple side-area compactness. "
                    "'april' = full April 24 baseline: loop sl=1..max_sl/epoch, density=0.3 fixed, "
                    "target_strength=0.8, slenderness + bbox_volume compactness.")
    return p.parse_args()


# ─────────────────────────────────────────────────────
#  Model factory
# ─────────────────────────────────────────────────────

def make_model(name, device):
    from src.models.baseline import AutoRegressiveBoxGenerator
    from src.models.atiss import CTATISSBoxGenerator
    from src.models.parallel_transformer import ParallelBoxGenerator
    from src.models.conditional_parallel import ConditionalParallelBoxGenerator
    return {
        "mlp": AutoRegressiveBoxGenerator,
        "ctatiss": CTATISSBoxGenerator,
        "parallel": ParallelBoxGenerator,
        "cond_parallel": ConditionalParallelBoxGenerator,
    }[name]().to(device)


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
            constraints.append((ovl.sum(dim=[1, 2]) / target_volumes.sum(dim=1)).unsqueeze(-1))

        return torch.cat(constraints, dim=-1)
    return cfn


# ─────────────────────────────────────────────────────
#  Loss computation (shared across all modes)
# ─────────────────────────────────────────────────────

def compute_losses(bx_pred, target_volumes, latent_codes, z_guess,
                   seq_length, use_padpp, target_adj=None, compactness_kind="simple"):
    device = bx_pred.device
    lwh, xyz = bx_pred[:, :, :3], bx_pred[:, :, 3:]
    internals = BatchInternals(xyz, lwh)
    L = {}

    # Constraints
    overlap_matrices = internals.checkOverlap()
    L["overlap"] = (overlap_matrices.norm(p=2, dim=1).mean(dim=1) / target_volumes.norm(p=2, dim=1)).mean()
    L["volume"] = ((internals.getBoxVolumes() - target_volumes) / target_volumes).norm(p=2)

    # Compactness
    if compactness_kind == "april":
        # April 24 baseline mix: slenderness + bbox-volume fill ratio
        frontal_area = internals.getRoughViewArea(view="frontal")
        side_view_area = internals.getRoughViewArea(view="side")
        top_view_area = internals.getRoughViewArea(view="top")
        bbox_volume = internals.getRoughVolume() / target_volumes.sum(dim=1)
        slenderness = ((frontal_area / (0.5 * side_view_area + 0.5 * top_view_area)) ** 0.5).mean()
        L["compactness"] = slenderness * 0.6 + 0.4 * bbox_volume.mean()
    else:
        norm = target_volumes.sum(dim=1) ** (2/3)
        L["compactness"] = (internals.getRoughViewArea(view="side") / norm).mean()

    # Quality for PA-DPP
    q = None
    if use_padpp:
        bbox_vol = internals.getRoughVolume() / target_volumes.sum(dim=1)
        q = 1.0 / bbox_vol.clamp(min=0.1)

    # DPP
    if seq_length > 1:
        ar = torch.log(torch.stack([lwh[:, :, 1] / lwh[:, :, 0], lwh[:, :, 2] / lwh[:, :, 0]], dim=2))
        Ax, _, Az = internals.getAdjacency()
        L["dpp_ar"] = PA_DPP_AR_loss(ar, q) if q is not None else DPP_AR_loss(ar)
        ldv_x = PA_DPP_loss_adjacency(Ax, q) if q is not None else DPP_loss_adjacency(Ax)
        ldv_z = PA_DPP_loss_adjacency(Az, q) if q is not None else DPP_loss_adjacency(Az)
        L["ldv"] = ldv_x + ldv_z
    else:
        L["dpp_ar"] = torch.tensor(0.0, device=device)
        L["ldv"] = torch.tensor(0.0, device=device)

    # MI
    L["mi"] = mi_loss(latent_codes, z_guess)

    # Adjacency conformity (conditional model only)
    if target_adj is not None:
        from src.constraints import constraint_loss
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

    from src.models.parallel_transformer import ParallelBoxGenerator

    # Defaults
    kwargs = dict(
        d_model=256, n_layers=4,
        z_conditioning="adaln", vol_encoding="fourier", vol_concat_head=True,
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
    elif args.ablation == "no_mi":
        pass  # handled in loss computation, not architecture
    else:
        raise ValueError(f"unknown ablation: {args.ablation}")

    return ParallelBoxGenerator(**kwargs).to(device)

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
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
    use_padpp = args.dpp == "padpp"
    use_mi = args.ablation != "no_mi"



    wandb.init(project="box-arrangement", name=args.run_name or None, config=vars(args))

    # Mode-specific setup
    if use_proj:
        from src.projector import lm_project_step, gated_ema_update
        from src.config import projector_config as pcfg, objective_weights as ow
        lambda_c = torch.zeros(2, device=device)
        lambda_d = torch.zeros(1, device=device)
        grad_clip = pcfg["grad_clip"]
    else:
        from src.config import get_loss_weights
        grad_clip = 1.0

    for epoch in tqdm(range(args.epochs), ncols=100):
        optimizer.zero_grad()
        z = torch.randn(args.batch_size, latent_dim, device=device)

        loop_all_sl = is_ar or (is_cond and args.cond_recipe in ("april_unif", "april"))
        seq_lengths = range(1, args.max_seq_length + 1) if loop_all_sl else \
                      [torch.randint(1, args.max_seq_length + 1, (1,)).item()]

        total_obj = torch.tensor(0.0, device=device)
        total_con = torch.tensor(0.0, device=device)
        total_disp = torch.tensor(0.0, device=device)
        vol_posts, ovl_posts, disp_posts = [], [], []

        for sl in seq_lengths:
            vols = (volume_limits[0] + torch.rand(args.batch_size, sl, device=device)
                    * (volume_limits[1] - volume_limits[0]))

            # Forward
            tgt_adj = None
            if is_cond:
                from src.constraints import sample_target
                if args.cond_recipe == "april_unif":
                    density = torch.rand(args.batch_size, device=device) * 0.5
                    tgt_strength = 0.8
                elif args.cond_recipe == "april":
                    density = 0.3
                    tgt_strength = 0.8
                else:
                    density = 0.3
                    tgt_strength = 0.3
                tgt_adj = sample_target(args.batch_size, sl, density=density,
                                        target_strength=tgt_strength, device=device)
                bx, zg = model(vols, z, target=tgt_adj)
            else:
                bx, zg = model(vols, z)

            compactness_kind = "april" if (is_cond and args.cond_recipe == "april") else "simple"
            L = compute_losses(bx, vols, z, zg, sl, use_padpp, tgt_adj,
                               compactness_kind=compactness_kind)

            # ── Projector path ────────────────────────────
            if use_proj:
                try:
                    y_hat = bx.view(args.batch_size, -1)
                    _, c_pre, c_post, disp = lm_project_step(
                        y_hat, make_constraint_fn(vols, sl),
                        delta=pcfg["delta"], lam_min=pcfg["lam_min"])

                    cr = c_pre.abs()
                    total_con = total_con + (lambda_c[0].detach() * cr[:, :sl]).sum(-1).mean()
                    if sl > 1:
                        total_con = total_con + (lambda_c[1].detach() * cr[:, sl:]).sum(-1).mean()

                    dm = disp.pow(2).sum(-1).add(1e-12).sqrt()
                    total_disp = total_disp + (lambda_d.detach() * dm).mean()

                    with torch.no_grad():
                        cp = c_post.abs()
                        vol_posts.append(cp[:, :sl].mean())
                        if sl > 1: ovl_posts.append(cp[:, sl:].mean())
                        disp_posts.append(dm.detach().mean())
                except Exception as e:
                    tqdm.write(f"Projector failed sl={sl}: {e}")

                total_obj = total_obj + (
                    ow["compactness"] * L["compactness"]
                    + ow["dpp_aspect_ratios"] * L["dpp_ar"]
                    + ow["ldv"] * L["ldv"]
                    + (ow["infonce"] * L["mi"] if use_mi else 0.0)
                    + ow.get("conform", 30.0) * L.get("conform", 0.0)
                )

            # ── Ramp path ─────────────────────────────────
            else:
                wts = get_loss_weights(epoch)
                λo, λv, λc, λl, λa, _, λm = wts[:7]
                λcf = wts[7] if len(wts) > 7 else 0.0
                total_obj = total_obj + (
                    λo * L["overlap"] + λv * L["volume"]
                    + λc * L["compactness"] + λl * L["ldv"]
                    + λa * L["dpp_ar"]
                    + (λm * L["mi"] if use_mi else 0.0)
                    + λcf * L.get("conform", 0.0)
                )

        # ── Step ──────────────────────────────────────────
        total_loss = total_obj + total_con + total_disp
        if torch.isnan(total_loss):
            tqdm.write(f"NaN at epoch {epoch}"); break

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        # ── Projector EMA ─────────────────────────────────
        if use_proj and vol_posts:
            with torch.no_grad():
                lambda_c[0] = gated_ema_update(lambda_c[0], torch.stack(vol_posts).mean(),
                                               pcfg["feas_tol"], pcfg["ema_max_decades"], pcfg["ema_rate"])
                if ovl_posts:
                    lambda_c[1] = gated_ema_update(lambda_c[1], torch.stack(ovl_posts).mean(),
                                                   pcfg["feas_tol"], pcfg["ema_max_decades"], pcfg["ema_rate"])
                if disp_posts:
                    lambda_d[0] = gated_ema_update(lambda_d[0], torch.stack(disp_posts).mean(),
                                                   pcfg["disp_tol"], pcfg["ema_max_decades"], pcfg["ema_rate"])

        # ── Log ───────────────────────────────────────────
        ld = {"total_loss": total_loss.item()}
        if use_proj:
            ld.update({"con_loss": total_con.item(), "disp_loss": total_disp.item(),
                       "lam_c_vol": lambda_c[0].item(), "lam_c_ovl": lambda_c[1].item(),
                       "lam_d": lambda_d[0].item()})
        wandb.log(ld, step=epoch)

        if (epoch + 1) % 100 == 0:
            tqdm.write(f"[{epoch+1}] loss={total_loss.item():.3f}")

    # ── Save ──────────────────────────────────────────────
    suffix = f"_{args.ablation}" if args.ablation != "baseline" else ""
    if is_cond and args.cond_recipe != "default":
        suffix = f"{suffix}_{args.cond_recipe}"
    name = f"{args.model}_{args.training}_{args.dpp}{suffix}_s{args.seed}.pth"
    path = os.path.join(os.path.dirname(__file__), "experiments", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved: {path}")
    wandb.finish()


if __name__ == "__main__":
    train(parse_args())
