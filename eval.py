import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import latent_dim, volume_limits
from src.constraints import sample_target, satisfaction_rate
from src.geometry.internals import BatchInternals
from src.inference import batch_optimized_inference
from src.losses import DPP_AR_loss, DPP_loss, DPP_loss_adjacency, mi_loss
from src.models.atiss import CTATISSBoxGenerator
from src.models.baseline import AutoRegressiveBoxGenerator
from src.models.causal_transformer import CausalTransformerBoxGenerator
from src.models.conditional_parallel import ConditionalParallelBoxGenerator
from src.models.parallel_transformer import ParallelBoxGenerator

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


MODEL_CLASSES = {
    "ct": CausalTransformerBoxGenerator,
    "ctatiss": CTATISSBoxGenerator,
    "mlp": AutoRegressiveBoxGenerator,
    "parallel": ParallelBoxGenerator,
    "cond_parallel": ConditionalParallelBoxGenerator,
}

DEFAULT_WEIGHTS = "experiments/conditional_parallel.pth"
DEFAULT_MODEL = "cond_parallel"
DEFAULT_SEQ_LEN = 3
DEFAULT_N = 100
DEFAULT_NUM_SEEDS = 10
DEFAULT_OUTPUT_DIR = "smoothness/conditional"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate box arrangement model")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, choices=MODEL_CLASSES.keys()
    )
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--num_seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot_name", type=str, default="")
    parser.add_argument("--skip_smoothness", action="store_true")
    parser.add_argument("--conditional", type=bool, default=True)
    return parser.parse_args()


def sample_volumes(n, seq_len):
    return volume_limits[0] + torch.rand(n, seq_len) * (
        volume_limits[1] - volume_limits[0]
    )


def run_model(model, z, volumes):
    with torch.no_grad():
        boxes, _ = model(volumes, z)
    lwh = boxes[:, :, :3]
    xyz = boxes[:, :, 3:]
    return xyz, lwh


def evaluate_seed(model, seed, N, seq_len, conditional=False, density=0.3):
    torch.manual_seed(seed)

    z_samples = torch.randn(N, latent_dim, device=device)
    volumes = sample_volumes(N, seq_len).to(device)
    xyz, lwh = run_model(model, z_samples, volumes)

    # geometry
    internals = BatchInternals(xyz, lwh)
    overlap = internals.checkOverlap().sum(dim=[1, 2]).mean().item()
    actual_vols = internals.getBoxVolumes()
    vol_accuracy = 1 - ((actual_vols - volumes).abs() / volumes).mean().item()
    rough_vol = internals.getRoughVolume()
    fill_ratio = (actual_vols.sum(dim=1) / rough_vol).mean().item()

    targets = None
    if conditional:
        targets = sample_target(N, seq_len, density=density, device=device)

    # DPP diversity
    with torch.no_grad():
        if conditional:
            boxes_batch, z_pred = model(volumes, z_samples, target=targets)
        else:
            boxes_batch, z_pred = model(volumes, z_samples)
    dpp_scores = []
    for _ in range(20):
        idx = torch.randperm(N)[:35]
        score = -DPP_loss(boxes_batch[idx], sigma=0.5).item()
        dpp_scores.append(score)

    # MI
    with torch.no_grad():
        if conditional:
            boxes_eval, z_pred = model(volumes, z_samples, target=targets)
        else:
            boxes_eval, z_pred = model(volumes, z_samples)
    mi = mi_loss(z_samples, z_pred).item()

    # LDV
    lwh_e = boxes_eval[:, :, :3]
    xyz_e = boxes_eval[:, :, 3:]
    internals_e = BatchInternals(xyz_e, lwh_e)
    Ax, Ay, Az = internals_e.getAdjacency()
    ldv = (DPP_loss_adjacency(Ax) + DPP_loss_adjacency(Az)).item()

    # aspect ratio DPP
    ar_dpp = 0.0
    if seq_len > 1:
        aspect_ratios = torch.log(
            torch.stack(
                [lwh_e[:, :, 1] / lwh_e[:, :, 0], lwh_e[:, :, 2] / lwh_e[:, :, 0]],
                dim=2,
            )
        )
        ar_dpp = DPP_AR_loss(aspect_ratios).item()

    result = {
        "overlap": overlap,
        "vol_accuracy": vol_accuracy,
        "compactness": fill_ratio,
        "dpp": np.mean(dpp_scores),
        "mi": mi,
        "ldv": ldv,
        "ar_dpp": ar_dpp,
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
        return "FAILED — no gradient"
    elif torch.isnan(z_grad.grad).any():
        return "FAILED — NaN gradient"
    else:
        return f"OK — grad norm {z_grad.grad.norm().item():.4f}"


def run_smoothness(model, seq_len, output_dir, plot_name):
    n_points = 50
    latent_values = torch.linspace(-3.5, 3.5, n_points)

    torch.manual_seed(0)
    vols_fixed = sample_volumes(1, seq_len).to(device).expand(n_points, -1)

    # Raw model output
    fig_x, axes_x = plt.subplots(4, seq_len, figsize=(seq_len * 3.5, 10))
    fig_z, axes_z = plt.subplots(4, seq_len, figsize=(seq_len * 3.5, 10))

    for latent_idx in range(4):
        z_sweep = torch.zeros(n_points, latent_dim, device=device)
        z_sweep[:, latent_idx] = latent_values
        xyz_s, _ = run_model(model, z_sweep, vols_fixed)

        for box_idx in range(seq_len):
            axes_x[latent_idx, box_idx].plot(
                latent_values.numpy(), xyz_s[:, box_idx, 0].cpu().numpy()
            )
            axes_x[latent_idx, box_idx].set_xlabel(f"z[{latent_idx}]")
            axes_x[latent_idx, box_idx].set_ylabel(f"box {box_idx} x pos")

            axes_z[latent_idx, box_idx].plot(
                latent_values.numpy(), xyz_s[:, box_idx, 2].cpu().numpy()
            )
            axes_z[latent_idx, box_idx].set_xlabel(f"z[{latent_idx}]")
            axes_z[latent_idx, box_idx].set_ylabel(f"box {box_idx} z pos")

    fig_x.tight_layout()
    fig_x.savefig(f"{output_dir}/smoothness_x_{seq_len}{plot_name}.png", dpi=300)
    fig_z.tight_layout()
    fig_z.savefig(f"{output_dir}/smoothness_z_{seq_len}{plot_name}.png", dpi=300)
    plt.close(fig_x)
    plt.close(fig_z)

    # After cleanup
    fig_x, axes_x = plt.subplots(4, seq_len, figsize=(seq_len * 3.5, 10))
    fig_z, axes_z = plt.subplots(4, seq_len, figsize=(seq_len * 3.5, 10))

    for latent_idx in range(4):
        z_sweep = torch.zeros(n_points, latent_dim, device=device)
        z_sweep[:, latent_idx] = latent_values
        with torch.no_grad():
            xyz_clean, _ = batch_optimized_inference(model, vols_fixed, z_sweep)

        for box_idx in range(seq_len):
            axes_x[latent_idx, box_idx].plot(
                latent_values.numpy(), xyz_clean[:, box_idx, 0].detach().cpu().numpy()
            )
            axes_x[latent_idx, box_idx].set_xlabel(f"z[{latent_idx}]")
            axes_x[latent_idx, box_idx].set_ylabel(f"box {box_idx} x pos")

            axes_z[latent_idx, box_idx].plot(
                latent_values.numpy(), xyz_clean[:, box_idx, 2].detach().cpu().numpy()
            )
            axes_z[latent_idx, box_idx].set_xlabel(f"z[{latent_idx}]")
            axes_z[latent_idx, box_idx].set_ylabel(f"box {box_idx} z pos")

    fig_x.tight_layout()
    fig_x.savefig(
        f"{output_dir}/smoothness_cleanup_x_{seq_len}{plot_name}.png", dpi=300
    )
    fig_z.tight_layout()
    fig_z.savefig(
        f"{output_dir}/smoothness_cleanup_z_{seq_len}{plot_name}.png", dpi=300
    )
    plt.close(fig_x)
    plt.close(fig_z)


def run_density_sweep(model, seq_len, N, num_seeds, output_dir, plot_name):
    densities = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    results_by_density = {}

    for density in densities:
        seed_results = []
        for seed in range(num_seeds):
            r = evaluate_seed(
                model, seed, N, seq_len, conditional=True, density=density
            )
            seed_results.append(r)
        agg = {
            k: (
                np.mean([r[k] for r in seed_results]),
                np.std([r[k] for r in seed_results]),
            )
            for k in seed_results[0].keys()
        }
        results_by_density[density] = agg
        print(
            f"density={density:.1f}  "
            f"sat={agg['satisfaction'][0]:.3f}±{agg['satisfaction'][1]:.3f}  "
            f"overlap={agg['overlap'][0]:.4f}  "
            f"vol_acc={agg['vol_accuracy'][0]:.3f}  "
            f"fill={agg['compactness'][0]:.3f}"
        )

    # Plot: satisfaction and overlap vs density
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    sats = [results_by_density[d]["satisfaction"][0] for d in densities]
    sat_err = [results_by_density[d]["satisfaction"][1] for d in densities]
    overlaps = [results_by_density[d]["overlap"][0] for d in densities]

    ax1.errorbar(
        densities,
        sats,
        yerr=sat_err,
        marker="o",
        color="C0",
        label="Satisfaction rate",
    )
    ax1.set_xlabel("Constraint density")
    ax1.set_ylabel("Satisfaction rate", color="C0")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2.plot(densities, overlaps, marker="s", color="C3", label="Overlap")
    ax2.set_ylabel("Overlap", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    plt.title(f"Density sweep (seq_len={seq_len}, N={N}, {num_seeds} seeds)")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/density_sweep_{seq_len}{plot_name}.png", dpi=300)
    plt.close(fig)

    return results_by_density


# -------------------------------------------------
#  Main
# -------------------------------------------------


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    conditional = args.conditional
    # model
    ModelClass = MODEL_CLASSES[args.model]
    model = ModelClass().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # multi-seed evaluation
    all_results = []
    for seed in range(args.num_seeds):
        result = evaluate_seed(model, seed, args.n, args.seq_len)
        all_results.append(result)

    print(
        f"\nResults over {args.num_seeds} seeds (N={args.n}, seq_len={args.seq_len}):"
    )
    print(f"Weights: {args.weights}")
    print("-" * 50)
    for key in all_results[0].keys():
        values = [r[key] for r in all_results]
        print(f"{key:18s}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # Gradient flow (once)
    print(f"\nGradient flow:      {check_gradient_flow(model, args.seq_len)}")

    # Smoothness plots
    if not args.skip_smoothness:
        print("\nGenerating smoothness plots...")
        run_smoothness(model, args.seq_len, args.output_dir, args.plot_name)
        print(f"Saved to {args.output_dir}/")

    print("\nRunning density sweep...")

    run_density_sweep(
        model,
        args.seq_len,
        args.n,
        args.num_seeds,
        args.output_dir,
        args.plot_name,
    )
    print(f"Saved density sweep to {args.output_dir}/")
