import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Slider

# Add parent directory to path for UAV imports
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(
    os.path.dirname(script_dir)
)  # Go up two levels from subsystems/
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.config import latent_dim, volume_limits
from src.geometry.internals import Internals
from src.inference import batch_optimized_inference
from src.models.atiss import CTATISSBoxGenerator
from src.models.baseline import AutoRegressiveBoxGenerator
from src.models.causal_transformer import CausalTransformerBoxGenerator

# Initialize model with same parameters as training
model = AutoRegressiveBoxGenerator()
# model = CausalTransformerBoxGenerator()
# model = CTATISSBoxGenerator()

model_path = os.path.join(os.path.dirname(__file__), "experiments/MLP_v1_seqlen10.pth")
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    print("Loaded pretrained model from", model_path)
else:
    print("Warning: Could not find pretrained model at", model_path)
model.eval()  # Set to evaluation mode

# Initial values
n_boxes = 3
volumes = torch.ones(1, n_boxes) * 0.05
latent_z = torch.zeros(1, latent_dim)

# Create figure and subplots
fig = plt.figure(figsize=(20, 10))
volume_text = fig.text(0.02, 0.62, "", fontsize=10)
overlap_text = fig.text(0.02, 0.6, "", fontsize=10)
overlap_matrix_text = fig.text(
    0.02, 0.8, "", fontsize=10
)  # New text element for overlap matrix
adjacency_text = fig.text(
    0.82, 0.05, "", fontsize=10
)  # New text element for adjacency matrices

# Create main 3D plot in the center
ax3d = fig.add_axes((0.25, 0.1, 0.5, 0.8), projection="3d")
ax3d.set_xlabel("X (m)", fontsize=12)
ax3d.set_ylabel("Y (m)", fontsize=12)
ax3d.set_zlabel("Z (m)", fontsize=12)
ax3d.set_facecolor("white")

# Calculate slider positions
slider_width = 0.15
slider_height = 0.02
gap = 0.01

# Left side - volume sliders
volume_x = 0.05
volume_sliders = []
for i in range(n_boxes):
    ax = plt.axes(
        (volume_x, 0.55 - i * (slider_height + gap), slider_width, slider_height)
    )
    slider = Slider(
        ax, f"Vol {i + 1}", volume_limits[0], volume_limits[1], valinit=0.05
    )
    volume_sliders.append(slider)

# Right side - n_boxes and latent sliders
right_x = 0.8
# N boxes slider at the top right
ax_n_boxes = plt.axes((right_x, 0.8, slider_width, slider_height))
s_n_boxes = Slider(ax_n_boxes, "N Boxes", 1, 10, valinit=n_boxes, valstep=1)

# Latent sliders below n_boxes
latent_sliders = []
for i in range(latent_dim):
    ax = plt.axes(
        [right_x, 0.7 - i * (slider_height + gap), slider_width, slider_height]
    )
    slider = Slider(ax, f"z{i}", -3.0, 3.0, valinit=0.0)
    latent_sliders.append(slider)


def update(val):
    global n_boxes, volumes, latent_z, volume_sliders, latent_sliders
    ax3d.clear()
    ax3d.set_proj_type("ortho")  # Set orthographic projection

    # Set plot appearance
    ax3d.set_facecolor("white")
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.grid(False)
    ax3d.set_xlabel("X (m)", fontsize=12)
    ax3d.set_ylabel("Y (m)", fontsize=12)
    ax3d.set_zlabel("Z (m)", fontsize=12)
    ax3d.set_proj_type("ortho")

    # Update volumes from sliders
    for i, slider in enumerate(volume_sliders[:n_boxes]):
        volumes[0, i] = slider.val

    # Update latent vector from sliders
    for i, slider in enumerate(latent_sliders):
        latent_z[0, i] = slider.val

    # Generate box arrangements
    with torch.no_grad():
        print("\nINFERENCE with interactive_viz.py")
        print("\nVolumes:")
        print(volumes)
        print("\nLatent vector z:")
        print(latent_z)

        print("\nGenerating box arrangement...")
        bx_pred, _ = model.forward(volumes, latent_z)
        lwh = bx_pred[:, :, :3]
        xyz = bx_pred[:, :, 3:]

        print("\nOutput box dimensions (length, width, height):")
        print(lwh)
        print("\nOutput box positions (x, y, z):")
        print(xyz)

        # Run optimized inference
        print("\nRunning optimized inference...")
        xyz_opt_batch, lwh_opt_batch = batch_optimized_inference(
            model, volumes, latent_z
        )

    # Create Internals objects and plot boxes
    internals = Internals(xyz[0].numpy(), lwh[0].numpy())
    optimized_internals = Internals(
        xyz_opt_batch[0].detach().cpu().numpy(), lwh_opt_batch[0].detach().cpu().numpy()
    )

    internals.checkOverlap()
    optimized_internals.checkOverlap()

    # Plot original boxes with lower alpha
    internals.getBoxSurfaces(ax3d, np.array([0, 0, 0]), alpha=0.1)
    # Plot optimized boxes with higher alpha
    optimized_internals.getBoxSurfaces(ax3d, np.array([0, 0, 0]), alpha=1)

    # Calculate metrics for both layouts
    orig_frontal_area = internals.getViewArea("frontal")
    orig_rough_frontal_area = internals.getRoughViewArea("frontal")
    orig_rough_volume = internals.getRoughVolume()
    orig_overlap = internals.checkOverlap().mean()

    opt_frontal_area = optimized_internals.getViewArea("frontal")
    opt_rough_frontal_area = optimized_internals.getRoughViewArea("frontal")
    opt_rough_volume = optimized_internals.getRoughVolume()
    opt_overlap = optimized_internals.checkOverlap().mean()

    # Display metrics
    area_text = (
        f"Convex Hull areas [m²]:\n"
        f"Original: frontal={orig_frontal_area:.3f} (rough={orig_rough_frontal_area:.3f})\n"
        f"Optimized: frontal={opt_frontal_area:.3f} (rough={opt_rough_frontal_area:.3f})"
    )
    ax3d.set_title(area_text, pad=20, fontsize=10)

    # Update volume text with both original and optimized values
    volume_text_str = (
        f"Rough Volume [m³]:\n"
        f"Original: {orig_rough_volume:.3f}\n"
        f"Optimized: {opt_rough_volume:.3f}\n\n"
        f"Box Volumes [m³]:\n"
    )

    # Get convex hull points from side view and scatter plot them
    if False:
        hull_points = internals.getConvexHullPoints("side")
        x_coords = [p[0] for p in hull_points]
        z_coords = [p[1] for p in hull_points]
        # Plot points at y=2 (slightly offset from the boxes for visibility)
        ax3d.scatter(
            x_coords,
            [2] * len(hull_points),
            z_coords,
            color="red",
            s=100,
            label="Convex Hull Points Side",
        )

    # Get convex hull points from front view and scatter plot them
    if False:
        hull_points = internals.getConvexHullPoints("frontal")
        y_coords = [p[0] for p in hull_points]
        z_coords = [p[1] for p in hull_points]
        # Plot points at y=2 (slightly offset from the boxes for visibility)
        ax3d.scatter(
            [2] * len(hull_points),
            y_coords,
            z_coords,
            color="black",
            s=100,
            label="Convex Hull Points Frontal",
        )

    orig_box_volumes = internals.getBoxVolumes()
    opt_box_volumes = optimized_internals.getBoxVolumes()
    for i, (orig_vol, opt_vol) in enumerate(zip(orig_box_volumes, opt_box_volumes)):
        volume_text_str += f"Box {i + 1}: {orig_vol:.3f} → {opt_vol:.3f}\n"
    volume_text.set_text(volume_text_str)

    # Update overlap text with both values
    overlap_text.set_text(
        f"Overlap:\nOriginal: {orig_overlap:.3f}\nOptimized: {opt_overlap:.3f}"
    )

    # Add overlap matrix display for optimized layout
    overlap_matrix = optimized_internals.checkOverlap()
    overlap_matrix_str = "Overlap Matrix (Optimized):\n"
    for i in range(overlap_matrix.shape[0]):
        row = " ".join([f"{val:.2f}" for val in overlap_matrix[i]])
        overlap_matrix_str += f"{row}\n"
    overlap_matrix_text.set_text(overlap_matrix_str)

    # Add adjacency matrices display for optimized layout
    Ax, Ay, Az = optimized_internals.getAdjacency()
    adjacency_str = "Adjacency Matrices (Optimized):\n\nX-axis:\n"
    for i in range(Ax.shape[0]):
        row = " ".join([f"{val:.2f}" for val in Ax[i]])
        adjacency_str += f"{row}\n"

    adjacency_str += "\nY-axis:\n"
    for i in range(Ay.shape[0]):
        row = " ".join([f"{val:.2f}" for val in Ay[i]])
        adjacency_str += f"{row}\n"

    adjacency_str += "\nZ-axis:\n"
    for i in range(Az.shape[0]):
        row = " ".join([f"{val:.2f}" for val in Az[i]])
        adjacency_str += f"{row}\n"

    adjacency_text.set_text(adjacency_str)

    # Instead of fixed limits, set equal aspect ratio to ensure same scale
    ax3d.set_box_aspect([1, 1, 1])  # Make the plot cube-shaped
    # Get the current limits
    x_lim = ax3d.get_xlim()
    y_lim = ax3d.get_ylim()
    z_lim = ax3d.get_zlim()
    # Find the maximum range across all dimensions
    max_range = max(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0])
    # Get the mid points
    x_mid = (x_lim[1] + x_lim[0]) * 0.5
    y_mid = (y_lim[1] + y_lim[0]) * 0.5
    z_mid = (z_lim[1] + z_lim[0]) * 0.5
    # Set new limits with equal scale
    ax3d.set_xlim(x_mid - max_range * 0.5, x_mid + max_range * 0.5)
    ax3d.set_ylim(y_mid - max_range * 0.5, y_mid + max_range * 0.5)
    ax3d.set_zlim(z_mid - max_range * 0.5, z_mid + max_range * 0.5)

    plt.draw()


def update_n_boxes(val):
    global n_boxes, volumes, volume_sliders

    # Clear old volume sliders
    for slider in volume_sliders:
        slider.ax.remove()
    volume_sliders.clear()

    # Update number of boxes and volumes tensor
    n_boxes = int(val)
    volumes = (
        torch.ones(1, n_boxes) * 0.05
    )  # Changed to 2D tensor [batch_size, seq_len]

    # Calculate new positions
    slider_width = 0.15
    slider_height = 0.02
    gap = 0.01
    volume_x = 0.05

    # Recreate volume sliders on the left side
    for i in range(n_boxes):
        ax = plt.axes(
            [volume_x, 0.55 - i * (slider_height + gap), slider_width, slider_height]
        )
        slider = Slider(
            ax, f"Vol {i + 1}", volume_limits[0], volume_limits[1], valinit=0.7
        )
        slider.on_changed(update)
        volume_sliders.append(slider)

    update(None)
    plt.draw()


# Connect update functions to sliders
s_n_boxes.on_changed(update_n_boxes)
for slider in volume_sliders + latent_sliders:
    slider.on_changed(update)

# Initial update
update(None)


def main():
    plt.show()


if __name__ == "__main__":
    main()
