import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path for UAV imports
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.geometry.internals import BatchInternals
from src.training_plots import plot_training_loss

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(5)

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
from src.models.causal_transformer import CausalTransformerBoxGenerator

# Define device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

assert volume_limits[0] > 0, (
    "Volume limits must be greater than 0 to avoid numerical instability (division by 0 in volume loss calc)"
)


def train_internals():

    print("Training on device:", device)

    total_loss_history = []
    volume_deficit_loss_history = []
    overlap_loss_history = []
    compactness_loss_history = []
    ldv_loss_history = []
    dpp_aspect_ratios_history = []
    infonce_loss_history = []
    loss_weights_history = []

    model = CausalTransformerBoxGenerator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs), ncols=100):
        optimizer.zero_grad()

        # Generate random latent codes from N(0,1)
        latent_codes = torch.randn(
            batch_size, latent_dim, device=device
        )  # [batch_size, latent_dim]

        overlap_loss = 0
        volume_deficit_loss = 0
        compactness_loss = 0
        ldv_loss = 0
        dpp_aspect_ratios = 0
        mi_loss_val = 0

        for seq_length in range(1, max_seq_length + 1):
            # Generate random volumes between volume_limits[0] and volume_limits[1]
            target_volumes = volume_limits[0] + torch.rand(
                batch_size, seq_length, device=device
            ) * (volume_limits[1] - volume_limits[0])
            target_volumes.requires_grad_(
                True
            )  # needed later to discourage ∂adj_loss/∂target_volumes

            # Forward pass
            bx_pred, latent_code_guess = model(
                target_volumes, latent_codes
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

            compactness_loss += 0.6 * slenderness + 0.4 * boundingbox_volume.mean()

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
            if seq_length > 1:
                Ax, Ay, Az = internals.getAdjacency()
                # Ay skipped for now, since y = 0 enforced for all boxes for the time being
                adjacent_diversity_x = DPP_loss_adjacency(Ax)
                adjacent_diversity_z = DPP_loss_adjacency(Az)
                ldv_loss += adjacent_diversity_x + adjacent_diversity_z

            # Compute mutual information loss for latent code prediction
            mi_loss_val += mi_loss(latent_codes, latent_code_guess)

        (
            λ_overlap,
            λ_volume,
            λ_compactness,
            λ_ldv,
            λ_dpp_aspect_ratios,
            λ_adj_vol_sensitivity,
            λ_infonce,
        ) = get_loss_weights(epoch)

        # Total loss
        total_loss = (
            λ_overlap * overlap_loss
            + λ_volume * volume_deficit_loss
            + λ_compactness * compactness_loss
            + λ_ldv * ldv_loss
            + λ_dpp_aspect_ratios * dpp_aspect_ratios
            + λ_infonce * mi_loss_val
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
                ]
            )
        )

        # Backpropagation
        total_loss.backward()
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
    torch.save(
        model.state_dict(),
        os.path.join(os.path.dirname(__file__), "experiments/ct_v3_seqlen10.pth"),
    )


if __name__ == "__main__":
    train_internals()
