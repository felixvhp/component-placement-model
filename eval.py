import matplotlib.pyplot as plt
import numpy as np
import torch

from src.geometry.internals import BatchInternals
from src.models.baseline import (
    AutoRegressiveBoxGenerator,
    DPP_AR_loss,
    DPP_loss,
    DPP_loss_adjacency,
    latent_dim,
    mi_loss,
    volume_limits,
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
seq_len = 3
N = 100

model = AutoRegressiveBoxGenerator().to(device)
model.load_state_dict(torch.load("box_arrangement_model.pth", map_location=device))
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
fig, axes = plt.subplots(4, 3, figsize=(12, 10))

for latent_idx in range(4):
    z_sweep = torch.zeros(n_points, latent_dim, device=device)
    z_sweep[:, latent_idx] = latent_values
    vols = sample_volumes(n_points, seq_len).to(device)
    xyz_s, _ = run_model(z_sweep, vols)

    for box_idx in range(3):
        axes[latent_idx, box_idx].plot(
            latent_values.numpy(), xyz_s[:, box_idx, 0].cpu().numpy()
        )
        axes[latent_idx, box_idx].set_xlabel(f"z[{latent_idx}]")
        axes[latent_idx, box_idx].set_ylabel(f"box {box_idx} x pos")

plt.tight_layout()
plt.savefig("smoothness.png", dpi=300)
print("Smoothness plot saved to smoothness.png")
