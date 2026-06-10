"""Per-batch loss computation, orchestrating the geometry/diversity losses into
the dict consumed by the training loop."""

import torch

from cpm.constraints import constraint_loss
from cpm.geometry.internals import BatchInternals
from cpm.losses import DPP_AR_loss, DPP_loss_adjacency, mi_loss


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
        Ax, Ay, Az = internals.getAdjacency()
        L["conform"] = constraint_loss(torch.stack([Ax, Ay, Az], dim=-1), target_adj)

    return L
