import os
import sys

import numpy as np
import torch
from tqdm import tqdm
import wandb

# Add parent directory to path for UAV imports
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.geometry.internals import BatchInternals
from src.training_plots import plot_training_loss
from src.constraints import sample_target, constraint_loss
from src.models.conditional_parallel import ConditionalParallelBoxGenerator

# Set random seeds for reproducibility (override via DIAG_SEED env var)
_DIAG_SEED = int(os.environ.get("DIAG_SEED", "42"))
torch.manual_seed(_DIAG_SEED)
np.random.seed(_DIAG_SEED // 8 + 1 if _DIAG_SEED != 42 else 5)

from src.config import (
    batch_size,
    epochs,
    get_loss_weights,
    latent_dim,
    lr,
    max_seq_length,
    volume_limits,
)
from src.losses import (
    DPP_AR_loss,
    DPP_loss_adjacency,
    mi_loss,
)
from src.models.atiss import CTATISSBoxGenerator
from src.models.causal_transformer import CausalTransformerBoxGenerator
from src.models.baseline import AutoRegressiveBoxGenerator
from src.models.parallel_transformer import ParallelBoxGenerator

# Define device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

DIAG_OUT = os.environ.get("DIAG_OUT", "experiments/debug/conditional_parallel_new13.pth")
DIAG_RUN = os.environ.get("DIAG_RUN", None)
DIAG_DENSITY = float(os.environ.get("DIAG_DENSITY", "0.3"))
DIAG_DENSITY_MAX = float(os.environ.get("DIAG_DENSITY_MAX", str(DIAG_DENSITY)))  # if > DIAG_DENSITY, sample density per-step from U(DIAG_DENSITY, DIAG_DENSITY_MAX)
DIAG_TGT_STRENGTH = float(os.environ.get("DIAG_TGT_STRENGTH", "0.3"))
DIAG_ALPHA_CONFORM = float(os.environ.get("DIAG_ALPHA_CONFORM", "1.0"))  # only affects conform_loss adjacency, NOT ldv
DIAG_LAMBDA_ADJ_SCALE = float(os.environ.get("DIAG_LAMBDA_ADJ_SCALE", "1.0"))  # multiplicative scaler on λ_adjacency from config

wandb.init(
    project="box-arrangement",
    name=DIAG_RUN,
    config={
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "max_seq_length": max_seq_length,
        "latent_dim": latent_dim,
        "volume_limits": volume_limits,
        "model": "ConditionalParallelBoxGenerator",
        "device": str(device),
        "out_path": DIAG_OUT,
        "diag_density": DIAG_DENSITY,
        "diag_density_max": DIAG_DENSITY_MAX,
        "diag_seed": _DIAG_SEED,
        "diag_target_strength": DIAG_TGT_STRENGTH,
        "diag_alpha_conform": DIAG_ALPHA_CONFORM,
        "diag_lambda_adj_scale": DIAG_LAMBDA_ADJ_SCALE,
    }
)

assert volume_limits[0] > 0, (
    "Volume limits must be greater than 0 to avoid numerical instability (division by 0 in volume loss calc)"
)


def train_internals():

    print("Training on device:", device)
    # print("GPU:", torch.cuda.get_device_name(0))

    total_loss_history = []
    volume_deficit_loss_history = []
    overlap_loss_history = []
    compactness_loss_history = []
    ldv_loss_history = []
    dpp_aspect_ratios_history = []
    infonce_loss_history = []
    loss_weights_history = []

    model = ConditionalParallelBoxGenerator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs), ncols=100):
        optimizer.zero_grad()

        # Generate random latent codes from N(0,1)
        latent_codes = torch.randn(
            batch_size, latent_dim
        ).to(device)  # [batch_size, latent_dim]

        overlap_loss = 0
        volume_deficit_loss = 0
        compactness_loss = 0
        ldv_loss = 0
        dpp_aspect_ratios = 0
        mi_loss_val = 0
        conform_loss = 0

        for seq_length in range(1, max_seq_length + 1):
            # Generate random volumes between volume_limits[0] and volume_limits[1]
            w = max(0.2, (seq_length / max_seq_length) ** 2)

            target_volumes = (
                volume_limits[0] + torch.rand(batch_size, seq_length)
                * (volume_limits[1] - volume_limits[0])
            ).to(device)
            target_volumes.requires_grad_(True)

            _density = DIAG_DENSITY if DIAG_DENSITY_MAX == DIAG_DENSITY else float(np.random.uniform(DIAG_DENSITY, DIAG_DENSITY_MAX))
            target_adjacency = sample_target(batch_size, seq_length, density=_density, target_strength=DIAG_TGT_STRENGTH, device=device)

            # Forward pass
            bx_pred, latent_code_guess = model(
                target_volumes, latent_codes, target=target_adjacency
            )  # [batch_size, seq_len, box_dim]

            # instatiate internals class
            lwh = bx_pred[:, :, :3]
            xyz = bx_pred[:, :, 3:]

            internals = BatchInternals(xyz, lwh)

            # Compute overlap matrices and box volumes
            overlap_matrices = internals.checkOverlap()
            actual_volumes = internals.getBoxVolumes()  # [batch_size, max_boxes]

            overlap_loss += (
                overlap_matrices.norm(p=2, dim=1).mean(dim=1)
                / target_volumes.norm(p=2, dim=1)
            ).mean()  # overlap normalized by volumes of the boxes

            volume_deficit_loss += (
                (actual_volumes - target_volumes) / target_volumes
            ).norm(p=2)

            # Compute frontal area and volume --> incentivize compact layouts
            frontal_area = internals.getRoughViewArea(view="frontal")
            side_view_area = internals.getRoughViewArea(view="side")
            top_view_area = internals.getRoughViewArea(view="top")

            boundingbox_volume = internals.getRoughVolume() / target_volumes.sum(
                dim=1
            )  # inverse of fill ratio
            slenderness = (
                (frontal_area / (0.5 * side_view_area + 0.5 * top_view_area)) ** 0.5
            ).mean()

            compactness_loss += slenderness * 0.6 + 0.4 * boundingbox_volume.mean()
            # compactness_loss += side_view_area.mean()

            # Compute diversity loss in aspect ratios
            if seq_length > 1:
                aspect_ratios = torch.log(
                    torch.stack(
                        [lwh[:, :, 1] / lwh[:, :, 0], lwh[:, :, 2] / lwh[:, :, 0]],
                        dim=2,
                    )
                )  #  w/l and h/l
                dpp_aspect_ratios += DPP_AR_loss(
                    aspect_ratios
                )  # enable diverse box aspect ratios

            # Compute layout diversity loss
            Ax, Ay, Az = internals.getAdjacency()
            if seq_length > 1:
                # Ay skipped for now, since y = 0 enforced for all boxes for the time being
                adjacent_diversity_x = DPP_loss_adjacency(Ax)
                adjacent_diversity_z = DPP_loss_adjacency(Az)
                ldv_loss += (adjacent_diversity_x + adjacent_diversity_z)

            # Compute mutual information loss for latent code prediction
            mi_loss_val += mi_loss(latent_codes, latent_code_guess)

            if DIAG_ALPHA_CONFORM != 1.0:
                # Recompute adjacency with custom alpha for conform_loss only
                # (does NOT affect ldv_loss which still uses alpha=1.0 from getAdjacency).
                _diff = internals.centers.unsqueeze(2) - internals.centers.unsqueeze(1)
                _ss = torch.tanh(DIAG_ALPHA_CONFORM * _diff)
                measured = _ss
            else:
                measured = torch.stack([Ax, Ay, Az], dim=-1)
            c_loss = constraint_loss(measured, target_adjacency)
            conform_loss += c_loss

        (
            λ_overlap,
            λ_volume,
            λ_compactness,
            λ_ldv,
            λ_dpp_aspect_ratios,
            λ_adj_vol_sensitivity,
            λ_infonce,
            λ_adjacency,
        ) = get_loss_weights(epoch)
        λ_adjacency = λ_adjacency * DIAG_LAMBDA_ADJ_SCALE

        # Total loss
        total_loss = (
            λ_overlap * overlap_loss
            + λ_volume * volume_deficit_loss
            + λ_compactness * compactness_loss
            + λ_ldv * ldv_loss
            + λ_dpp_aspect_ratios * dpp_aspect_ratios
            + λ_infonce * mi_loss_val
            + λ_adjacency * conform_loss
        )

        total_loss_history.append(total_loss.item())
        volume_deficit_loss_history.append(volume_deficit_loss.item())
        overlap_loss_history.append(overlap_loss.item())
        compactness_loss_history.append(compactness_loss.item())
        ldv_loss_history.append(ldv_loss.item())
        dpp_aspect_ratios_history.append(dpp_aspect_ratios.item())
        infonce_loss_history.append(mi_loss_val.item())
        loss_weights_history.append(
            np.array(
                [
                    λ_overlap,
                    λ_volume,
                    λ_compactness,
                    λ_ldv,
                    λ_dpp_aspect_ratios,
                    λ_infonce,
                    λ_adjacency,
                ]
            )
        )

        wandb.log({
            "total_loss": total_loss.item(),
            "overlap_loss": overlap_loss.item(),
            "volume_deficit_loss": volume_deficit_loss.item(),
            "compactness_loss": compactness_loss.item(),
            "ldv_loss": ldv_loss.item(),
            "dpp_aspect_ratios": dpp_aspect_ratios.item(),
            "infonce_loss": mi_loss_val.item(),
            "conform_loss": conform_loss.item(),
        }, step=epoch)

        if torch.isnan(total_loss):
            print(f"NaN total_loss at epoch {epoch}", flush=True)
            break

        # Backpropagation
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


        nan_grad = False
        for name, p in model.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                print(f"NaN gradient in {name} at epoch {epoch}", flush=True)
                nan_grad = True
                break
        if nan_grad:
            break


        optimizer.step()

        if (epoch + 1) % 100 == 0:
            plot_training_loss(
                total_loss_history,
                overlap_loss_history,
                volume_deficit_loss_history,
                compactness_loss_history,
                ldv_loss_history,
                dpp_aspect_ratios_history,
                infonce_loss_history,
                loss_weights_history,
            )

    # Save the model
    out_path = DIAG_OUT
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.path.dirname(__file__), out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved checkpoint to {out_path}", flush=True)

    wandb.finish()


if __name__ == "__main__":
    train_internals()
