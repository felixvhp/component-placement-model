import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import latent_dim, volume_limits
from src.constraints import sample_target, satisfaction_rate
from src.geometry.internals import BatchInternals
from src.inference import batch_optimized_inference
from src.losses import DPP_AR_loss, DPP_loss_adjacency, DPP_loss_adjacency_normalized, mi_loss
from src.models.atiss import CTATISSBoxGenerator
from src.models.baseline import AutoRegressiveBoxGenerator
from src.models.conditional_parallel import ConditionalParallelBoxGenerator
from src.models.parallel_transformer import ParallelBoxGenerator
from src.models.conditional_atiss import ConditionalCTATISSBoxGenerator

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


MODEL_CLASSES = {
    "ctatiss": CTATISSBoxGenerator,
    "mlp": AutoRegressiveBoxGenerator,
    "parallel": ParallelBoxGenerator,
    "cond_parallel": ConditionalParallelBoxGenerator,
    "cond_atiss": ConditionalCTATISSBoxGenerator,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate box arrangement model")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=MODEL_CLASSES.keys())
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--cleanup", type=str, default="off", choices=["on", "off"])
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--skip_smoothness", action="store_true")
    parser.add_argument("--skip_density_sweep", action="store_true")
    parser.add_argument("--plot_name", type=str, default="")
    parser.add_argument("--ablation", type=str, default="baseline",
                    choices=["baseline", "z_token", "z_both", "linear_vol", "no_vol_concat",
                             "small", "large", "no_mi"])
    parser.add_argument("--ldv-raw", action="store_true",
                    help="Use DPP_loss_adjacency (non-normalized) instead of the default "
                         "DPP_loss_adjacency_normalized for the ldv metric.")
    return parser.parse_args()


def sample_volumes(n, seq_len):
    return volume_limits[0] + torch.rand(n, seq_len) * (
        volume_limits[1] - volume_limits[0]
    )


def evaluate_seed(model, seed, N, seq_len, cleanup="off", conditional=False, density=0.3, ldv_raw=False):
    torch.manual_seed(seed)

    z_samples = torch.randn(N, latent_dim, device=device)
    volumes = sample_volumes(N, seq_len).to(device)

    targets = None
    if conditional:
        targets = sample_target(N, seq_len, density=density, device=device)

    # Forward pass (raw model output)
    with torch.no_grad():
        if conditional:
            boxes_raw, z_pred = model(volumes, z_samples, target=targets)
        else:
            boxes_raw, z_pred = model(volumes, z_samples)
    lwh_raw = boxes_raw[:, :, :3]
    xyz_raw = boxes_raw[:, :, 3:]

    # Optionally apply inference-time cleanup (unrolled GD on positions)
    if cleanup == "on":
        xyz_final, lwh_final = batch_optimized_inference(
            model, volumes, z_samples,
            target=targets if conditional else None,
        )
        xyz_final = xyz_final.detach()
        lwh_final = lwh_final.detach()
    else:
        xyz_final, lwh_final = xyz_raw, lwh_raw

    # Geometry metrics
    internals = BatchInternals(xyz_final, lwh_final)
    overlap = internals.checkOverlap().sum(dim=[1, 2]).mean().item()
    actual_vols = internals.getBoxVolumes()
    vol_accuracy = 1 - ((actual_vols - volumes).abs() / volumes).mean().item()

    # Compactness (fill ratio: per-box side-view area sum / bbox side area)
    sum_box_side_areas = (lwh_final[..., 0] * lwh_final[..., 2]).sum(dim=1)
    bbox_side_area = internals.getRoughViewArea(view="side")
    compactness = (sum_box_side_areas / bbox_side_area.clamp(min=1e-6)).mean().item()

    # MI
    mi = mi_loss(z_samples, z_pred).item()

    # Adjacency / LDV (computed on final boxes)
    Ax, Ay, Az = internals.getAdjacency()
    ldv_fn = DPP_loss_adjacency if ldv_raw else DPP_loss_adjacency_normalized
    ldv = (ldv_fn(Ax) + ldv_fn(Az)).item()

    # Diversity metrics
    xyz_flat = xyz_final.reshape(N, -1)
    position_spread = xyz_flat.std(dim=0).mean().item()

    shape_spread = 0.0
    ar_dpp = 0.0
    if seq_len > 1:
        log_ar = torch.log(
            torch.stack(
                [lwh_final[:, :, 1] / lwh_final[:, :, 0],
                 lwh_final[:, :, 2] / lwh_final[:, :, 0]],
                dim=2,
            )
        )
        shape_spread = log_ar.std(dim=0).mean().item()
        ar_dpp = DPP_AR_loss(log_ar).item()

    result = {
        "overlap": overlap,
        "vol_accuracy": vol_accuracy,
        "compactness": compactness,
        "position_spread": position_spread,
        "shape_spread": shape_spread,
        "ar_dpp": ar_dpp,
        "ldv": ldv,
        "mi": mi,
    }

    if conditional:
        measured = torch.stack([Ax, Ay, Az], dim=-1)
        result["satisfaction"] = satisfaction_rate(measured, targets).item()

    return result


def check_gradient_flow(model, seq_len):
    z_grad = torch.randn(1, latent_dim, requires_grad=True, device=device)
    vol_grad = sample_volumes(1, seq_len).to(device)
    boxes_grad, _ = model(vol_grad, z_grad)
    lwh_g = boxes_grad[:, :, :3]
    xyz_g = boxes_grad[:, :, 3:]
    internals_g = BatchInternals(xyz_g, lwh_g)
    overlap_g = internals_g.checkOverlap().sum()
    overlap_g.backward()

    if z_grad.grad is None:
        return {"status": "FAILED", "reason": "no gradient", "grad_norm": None}
    elif torch.isnan(z_grad.grad).any():
        return {"status": "FAILED", "reason": "NaN gradient", "grad_norm": None}
    else:
        return {"status": "OK", "reason": None, "grad_norm": z_grad.grad.norm().item()}


def run_smoothness(model, seq_len, output_dir, plot_name, cleanup="off", stride=2,
                   latent_indices=(0,)):
    n_points = 50
    latent_values = torch.linspace(-3.5, 3.5, n_points)

    torch.manual_seed(0)
    vols_fixed = sample_volumes(1, seq_len).to(device).expand(n_points, -1)

    # Which boxes / latent coordinates to plot
    box_indices = list(range(0, seq_len, stride))
    n_cols = len(box_indices)
    latent_indices = list(latent_indices)
    n_rows = len(latent_indices)

    fig_x, axes_x = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.6),
                                 squeeze=False)
    fig_z, axes_z = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.6),
                                 squeeze=False)

    for row_idx, latent_idx in enumerate(latent_indices):
        z_sweep = torch.zeros(n_points, latent_dim, device=device)
        z_sweep[:, latent_idx] = latent_values

        if cleanup == "on":
            with torch.no_grad():
                xyz_s, _ = batch_optimized_inference(model, vols_fixed, z_sweep)
            xyz_s = xyz_s.detach()
        else:
            with torch.no_grad():
                boxes_s, _ = model(vols_fixed, z_sweep)
            xyz_s = boxes_s[:, :, 3:]

        for col_idx, box_idx in enumerate(box_indices):
            ax_x = axes_x[row_idx, col_idx]
            ax_x.plot(
                latent_values.numpy(), xyz_s[:, box_idx, 0].cpu().numpy()
            )
            ax_x.set_xlabel(f"z[{latent_idx}]", fontsize=16)
            ax_x.set_ylabel(f"box {box_idx} x pos", fontsize=16)
            ax_x.tick_params(axis="both", labelsize=13)

            ax_z = axes_z[row_idx, col_idx]
            ax_z.plot(
                latent_values.numpy(), xyz_s[:, box_idx, 2].cpu().numpy()
            )
            ax_z.set_xlabel(f"z[{latent_idx}]", fontsize=16)
            ax_z.set_ylabel(f"box {box_idx} z pos", fontsize=16)
            ax_z.tick_params(axis="both", labelsize=13)

    fig_x.tight_layout()
    fig_x.savefig(
        f"{output_dir}/smoothness_x_sl{seq_len}_cleanup-{cleanup}{plot_name}.png",
        dpi=300,
    )
    fig_z.tight_layout()
    fig_z.savefig(
        f"{output_dir}/smoothness_z_sl{seq_len}_cleanup-{cleanup}{plot_name}.png",
        dpi=300,
    )
    plt.close(fig_x)
    plt.close(fig_z)


def run_density_sweep(model, seq_len, N, num_seeds, output_dir, plot_name, cleanup="off", ldv_raw=False):
    densities = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    results_by_density = {}

    for density in densities:
        seed_results = []
        for seed in range(num_seeds):
            r = evaluate_seed(
                model, seed, N, seq_len,
                cleanup=cleanup, conditional=True, density=density,
                ldv_raw=ldv_raw,
            )
            seed_results.append(r)
        agg = {
            k: {
                "mean": float(np.mean([r[k] for r in seed_results])),
                "std": float(np.std([r[k] for r in seed_results])),
            }
            for k in seed_results[0].keys()
        }
        results_by_density[f"{density:.2f}"] = agg
        print(
            f"density={density:.1f}  "
            f"sat={agg['satisfaction']['mean']:.3f}±{agg['satisfaction']['std']:.3f}  "
            f"overlap={agg['overlap']['mean']:.4f}  "
            f"vol_acc={agg['vol_accuracy']['mean']:.3f}  "
            f"fill={agg['compactness']['mean']:.3f}"
        )

    # Plot satisfaction + overlap vs density
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    sats = [results_by_density[f"{d:.2f}"]["satisfaction"]["mean"] for d in densities]
    sat_err = [results_by_density[f"{d:.2f}"]["satisfaction"]["std"] for d in densities]
    overlaps = [results_by_density[f"{d:.2f}"]["overlap"]["mean"] for d in densities]

    ax1.errorbar(densities, sats, yerr=sat_err, marker="o", color="C0",
                 label="Satisfaction rate")
    ax1.set_xlabel("Constraint density")
    ax1.set_ylabel("Satisfaction rate", color="C0")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2.plot(densities, overlaps, marker="s", color="C3", label="Overlap")
    ax2.set_ylabel("Overlap", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    plt.title(f"Density sweep (seq_len={seq_len}, N={N}, {num_seeds} seeds, cleanup={cleanup})")
    fig.tight_layout()
    fig.savefig(
        f"{output_dir}/density_sweep_sl{seq_len}_cleanup-{cleanup}{plot_name}.png",
        dpi=300,
    )
    plt.close(fig)

    return results_by_density


# -------------------------------------------------
#  Main
# -------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    if args.model == "parallel" and args.ablation != "baseline":
        # Reconstruct with same kwargs as training
        from src.models.parallel_transformer import ParallelBoxGenerator
        if args.ablation == "z_token":
            model = ParallelBoxGenerator(z_conditioning="token").to(device)
        elif args.ablation == "z_both":
            model = ParallelBoxGenerator(z_conditioning="both").to(device)
        elif args.ablation == "linear_vol":
            model = ParallelBoxGenerator(vol_encoding="linear").to(device)
        elif args.ablation == "no_vol_concat":
            model = ParallelBoxGenerator(vol_concat_head=False).to(device)
        elif args.ablation == "small":
            model = ParallelBoxGenerator(d_model=128, n_layers=2).to(device)
        elif args.ablation == "large":
            model = ParallelBoxGenerator(d_model=512, n_layers=6).to(device)
        elif args.ablation == "no_mi":
            model = ParallelBoxGenerator().to(device)
        else:
            model = ParallelBoxGenerator().to(device)
    else:
        ModelClass = MODEL_CLASSES[args.model]
        model = ModelClass().to(device)

    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # Multi-seed evaluation
    all_results = []
    for seed in range(args.num_seeds):
        result = evaluate_seed(
            model, seed, args.n, args.seq_len,
            cleanup=args.cleanup, conditional=args.conditional,
            ldv_raw=args.ldv_raw,
        )
        all_results.append(result)

    # Aggregate
    agg = {
        key: {
            "mean": float(np.mean([r[key] for r in all_results])),
            "std": float(np.std([r[key] for r in all_results])),
        }
        for key in all_results[0].keys()
    }

    print(f"\nResults over {args.num_seeds} seeds (N={args.n}, seq_len={args.seq_len}, cleanup={args.cleanup}):")
    print(f"Weights: {args.weights}")
    print("-" * 50)
    for key, v in agg.items():
        print(f"{key:18s}: {v['mean']:.4f} ± {v['std']:.4f}")

    # Gradient flow
    grad = check_gradient_flow(model, args.seq_len)
    print(f"\nGradient flow: {grad}")

    # Smoothness plots
    if not args.skip_smoothness:
        print("\nGenerating smoothness plots...")
        run_smoothness(model, args.seq_len, args.output_dir,
                       args.plot_name, cleanup=args.cleanup)

    # Density sweep (conditional only)
    density_results = None
    if args.conditional and not args.skip_density_sweep:
        print("\nRunning density sweep...")
        density_results = run_density_sweep(
            model, args.seq_len, args.n, args.num_seeds,
            args.output_dir, args.plot_name, cleanup=args.cleanup,
            ldv_raw=args.ldv_raw,
        )

    # Dump JSON
    output = {
        "model": args.model,
        "weights": args.weights,
        "seq_len": args.seq_len,
        "n_samples": args.n,
        "num_seeds": args.num_seeds,
        "cleanup": args.cleanup,
        "conditional": args.conditional,
        "metrics": agg,
        "per_seed": all_results,
        "gradient_flow": grad,
        "density_sweep": density_results,
    }

    ablation_suffix = f"_{args.ablation}" if args.ablation != "baseline" else ""
    json_path = os.path.join(
        args.output_dir,
        f"results_{args.model}{ablation_suffix}_sl{args.seq_len}_cleanup-{args.cleanup}.json",
    )
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON: {json_path}")
