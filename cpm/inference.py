import torch

from .config import volume_limits
from .geometry.internals import BatchInternals
from .constraints import lm_project_step, make_constraint_fn

# -------------------------------------------------------------
#  Pytorch "unrolled" gradient descent
# -------------------------------------------------------------


def batch_optimized_inference(
    model,
    target_volumes,
    latent_codes,
    target=None,
    iterations=10,
    lr=0.1,
    max_grad_norm=1.0,
    method="projector",
):

    batch_size = target_volumes.shape[0]
    seq_length = target_volumes.shape[1]

    # assert target volumes is within volume limits and tensor with 3 dims
    assert len(target_volumes.shape) == 2, (
        "Target volumes must be a tensor with 2 dimensions but is "
        + str(target_volumes.shape)
    )
    assert torch.all(target_volumes >= volume_limits[0]) and torch.all(
        target_volumes <= volume_limits[1]
    ), "Target volumes must be within volume limits"

    tol = 5e-2
    assert torch.all(latent_codes >= -3.5 - tol) and torch.all(
        latent_codes <= 3.5 + tol
    ), (
        "Latent Code min: "
        + str(latent_codes.min())
        + " | Latent Code max: "
        + str(latent_codes.max())
    )

    with torch.enable_grad():
        # Forward pass on trained model
        if target is not None:
            bx_pred, _ = model(target_volumes, latent_codes, target=target)
        else:
            bx_pred, _ = model(target_volumes, latent_codes)
        lwh_fixed = bx_pred[:, :, :3].clone()  # [seq_len, 3] numpy array
        xyz_init = bx_pred[:, :, 3:].clone()  # [seq_len, 3] numpy array

        if method == "projector":
            y = bx_pred.view(batch_size, -1).requires_grad_(True)
            cfn = make_constraint_fn(target_volumes, seq_length)
            for i in range(iterations):
                y_new, c_pre, _, _ = lm_project_step(y, cfn)
                if c_pre.abs().max() < 1e-8:
                    break
                y = y_new.requires_grad_(True)
            boxes = y.detach().view(batch_size, seq_length, 6)
            return boxes[:, :, 3:], boxes[:, :, :3]

        else:
            xyz_opt = xyz_init.clone().requires_grad_(True)
            λ_overlap = 1.0

            assert xyz_init.shape == (batch_size, seq_length, 3), (
                "xyz_init must be a tensor with shape (batch_size, seq_length, 3) but is "
                + str(xyz_init.shape)
            )
            assert lwh_fixed.shape == (batch_size, seq_length, 3), (
                "lwh_fixed must be a tensor with shape (batch_size, seq_length, 3) but is "
                + str(lwh_fixed.shape)
            )

            # iterate on xyz to find final compact + non-overlapping layout
            for i in range(iterations):
                # Assess layout quality
                internals = BatchInternals(xyz_opt, lwh_fixed)
                overlaps = internals.checkOverlap()
                boundingbox_volume = internals.getRoughVolume()
                frontal_area = internals.getRoughViewArea(view="frontal")

                # Compute scalarized loss
                box_volumes = internals.getBoxVolumes()  # [batch_size, max_boxes]

                safe_overlap_norm = torch.sqrt((overlaps**2).sum(dim=1) + 1e-6)
                overlap_loss = (
                    safe_overlap_norm.mean() / box_volumes.norm(p=2, dim=1).mean()
                )  # overlap normalized by volumes of the boxes

                if torch.all(overlap_loss < 1e-6):
                    break  # exit the loop ~ if no overlap

                volume_loss = (
                    boundingbox_volume.pow(2).mean() / 100.0
                )  # Scale down volume loss
                area_loss = frontal_area.pow(2).mean() / 10.0  # Scale down area loss

                loss = λ_overlap * overlap_loss + volume_loss + area_loss

                grad = torch.autograd.grad(loss, xyz_opt, create_graph=True)[0]

                # Clip gradients to prevent explosion
                grad_norm = torch.norm(grad)
                if grad_norm > max_grad_norm:
                    grad = grad * (max_grad_norm / grad_norm)

                xyz_opt = xyz_opt - lr * grad

                # Increase overlap penalty
                λ_overlap *= 1.5

            return xyz_opt, lwh_fixed
