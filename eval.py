import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.geometry.internals import BatchInternals
from src.inference import batch_optimized_inference
from src.models.baseline import (
    DPP_AR_loss,
    DPP_loss,
    DPP_loss_adjacency,
    latent_dim,
    mi_loss,
    volume_limits,
)
from src.models.causal_transformer import CausalTransformerBoxGenerator

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

ModelClass = CausalTransformerBoxGenerator
# ModelClass = AutoRegressiveBoxGenerator
weights_path = "box_arrangement_model10_ct2.pth"
output_dir = "smoothness/ct"
plot_name = ""
os.makedirs(output_dir, exist_ok=True)
seq_len = 10
N = 100

model = ModelClass().to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()


def sample_volumes(n, seq_len):
    return volume_limits[0] + torch.rand(n, seq_len) * (
        volume_limits[1] - volume_limits[0]
    )


def run_model(z, volumes):
    with torch.no_grad():
        boxes, _ = model(volumes, z)
    lwh = boxes[:, :, :3]
    xyz = boxes[:, :, 3:]
    return xyz, lwh


# geometry metrics
z_samples = torch.randn(N, latent_dim, device=device)
volumes = sample_volumes(N, seq_len).to(device)
xyz, lwh = run_model(z_samples, volumes)

internals = BatchInternals(xyz, lwh)
overlap = internals.checkOverlap().sum(dim=[1, 2]).mean().item()
actual_vols = internals.getBoxVolumes()
vol_accuracy = 1 - ((actual_vols - volumes).abs() / volumes).mean().item()
rough_vol = internals.getRoughVolume()
fill_ratio = (actual_vols.sum(dim=1) / rough_vol).mean().item()

print(f"Overlap:          {overlap:.4f}")
print(f"Volume accuracy:  {vol_accuracy:.4f}")
print(f"Compactness:      {fill_ratio:.4f}")

# DPP
boxes_batch, _ = model(volumes, z_samples)
dpp_scores = []
for _ in range(20):
    idx = torch.randperm(N)[:35]
    score = -DPP_loss(boxes_batch[idx], sigma=0.5).item()
    dpp_scores.append(score)
print(f"DPP diversity:    {np.mean(dpp_scores):.4f} ± {np.std(dpp_scores):.4f}")


boxes_eval, z_pred = model(volumes, z_samples)
lwh_e = boxes_eval[:, :, :3]
xyz_e = boxes_eval[:, :, 3:]

# MI
mi = mi_loss(z_samples, z_pred).item()
print(f"MI loss:          {mi:.4f}")

# LDV
internals_e = BatchInternals(xyz_e, lwh_e)
Ax, Ay, Az = internals_e.getAdjacency()
ldv = (DPP_loss_adjacency(Ax) + DPP_loss_adjacency(Az)).item()
print(f"LDV diversity:    {ldv:.4f}")

# aspect ratio DPP
if seq_len > 1:
    aspect_ratios = torch.log(
        torch.stack(
            [lwh_e[:, :, 1] / lwh_e[:, :, 0], lwh_e[:, :, 2] / lwh_e[:, :, 0]],
            dim=2,
        )
    )
    ar_dpp = DPP_AR_loss(aspect_ratios).item()
    print(f"Aspect ratio DPP: {ar_dpp:.4f}")

# Gradient flow
z_grad = torch.randn(1, latent_dim, requires_grad=True, device=device)
vol_grad = sample_volumes(1, seq_len).to(device)
boxes_grad, _ = model(vol_grad, z_grad)
lwh_g = boxes_grad[:, :, :3]
xyz_g = boxes_grad[:, :, 3:]
internals_g = BatchInternals(xyz_g, lwh_g)
overlap_g = internals_g.checkOverlap().sum()
overlap_g.backward()

if z_grad.grad is None:
    print("Gradient flow:    FAILED — no gradient")
elif torch.isnan(z_grad.grad).any():
    print("Gradient flow:    FAILED — NaN gradient")
else:
    print(f"Gradient flow:    OK — grad norm {z_grad.grad.norm().item():.4f}")


# Smoothness analysis
n_points = 50
latent_values = torch.linspace(-3.5, 3.5, n_points)
vols_fixed = sample_volumes(1, seq_len).to(device).expand(n_points, -1)

fig_x, axes_x = plt.subplots(4, seq_len, figsize=(seq_len * 3.5, 10))
fig_z, axes_z = plt.subplots(4, seq_len, figsize=(seq_len * 3.5, 10))

for latent_idx in range(4):
    z_sweep = torch.zeros(n_points, latent_dim, device=device)
    z_sweep[:, latent_idx] = latent_values
    xyz_s, _ = run_model(z_sweep, vols_fixed)

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

# cleanup
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
fig_x.savefig(f"{output_dir}/smoothness_cleanup_x_{seq_len}{plot_name}.png", dpi=300)
fig_z.tight_layout()
fig_z.savefig(f"{output_dir}/smoothness_cleanup_z_{seq_len}{plot_name}.png", dpi=300)

# xyz_clean, lwh_clean = batch_optimized_inference(model, volumes, z_samples)

# internals_clean = BatchInternals(xyz_clean, lwh_clean)
# overlap_clean = internals_clean.checkOverlap().sum(dim=[1, 2]).mean().item()
# rough_vol_clean = internals_clean.getRoughVolume()
# fill_ratio_clean = (actual_vols.sum(dim=1) / rough_vol_clean).mean().item()

# print("\n--- After cleanup ---")
# print(f"Overlap:          {overlap_clean:.4f}")
# print(f"Compactness:      {fill_ratio_clean:.4f}")
