"""Faithful repro of train.py inner loop to isolate the crash."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch.manual_seed(42)
device = torch.device('cuda')
print('device:', device, torch.cuda.get_device_name(0))

from src.config import batch_size, latent_dim, volume_limits
from src.constraints import sample_target, constraint_loss
from src.geometry.internals import BatchInternals
from src.losses import DPP_AR_loss, DPP_loss_adjacency, mi_loss
from src.models.conditional_parallel import ConditionalParallelBoxGenerator

print('Building model...')
model = ConditionalParallelBoxGenerator().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
print('Model built.')

latent_codes = torch.randn(batch_size, latent_dim, device=device)
overlap_loss = 0; volume_deficit_loss = 0; compactness_loss = 0
ldv_loss = 0; dpp_aspect_ratios = 0; mi_loss_val = 0; conform_loss = 0

max_seq_length = 10
for seq_length in range(1, max_seq_length + 1):
    print(f'--- sl={seq_length} ---', flush=True)
    try:
        target_volumes = (volume_limits[0] + torch.rand(batch_size, seq_length, device=device)
                          * (volume_limits[1] - volume_limits[0]))
        target_volumes.requires_grad_(True)
        target_adjacency = sample_target(batch_size, seq_length, density=0.3, device=device)
        torch.cuda.synchronize(); print(f'  sample_target OK', flush=True)

        bx_pred, latent_code_guess = model(target_volumes, latent_codes, target=target_adjacency)
        torch.cuda.synchronize(); print(f'  forward OK', flush=True)

        lwh = bx_pred[:, :, :3]; xyz = bx_pred[:, :, 3:]
        internals = BatchInternals(xyz, lwh)

        overlap_matrices = internals.checkOverlap()
        actual_volumes = internals.getBoxVolumes()
        torch.cuda.synchronize(); print(f'  overlap+volumes OK', flush=True)

        overlap_loss = overlap_loss + (overlap_matrices.norm(p=2, dim=1).mean(dim=1)
                                        / target_volumes.norm(p=2, dim=1)).mean()
        volume_deficit_loss = volume_deficit_loss + ((actual_volumes - target_volumes) / target_volumes).norm(p=2)

        frontal_area = internals.getRoughViewArea(view='frontal')
        side_view_area = internals.getRoughViewArea(view='side')
        top_view_area = internals.getRoughViewArea(view='top')
        bb_vol = internals.getRoughVolume() / target_volumes.sum(dim=1)
        torch.cuda.synchronize(); print(f'  view areas + bb vol OK', flush=True)

        slend = ((frontal_area / (0.5 * side_view_area + 0.5 * top_view_area)) ** 0.5).mean()
        compactness_loss = compactness_loss + slend * 0.6 + 0.4 * bb_vol.mean()

        if seq_length > 1:
            ar = torch.log(torch.stack([lwh[:, :, 1] / lwh[:, :, 0], lwh[:, :, 2] / lwh[:, :, 0]], dim=2))
            dpp_aspect_ratios = dpp_aspect_ratios + DPP_AR_loss(ar)
            torch.cuda.synchronize(); print(f'  dpp_ar OK', flush=True)

        Ax, Ay, Az = internals.getAdjacency()
        if seq_length > 1:
            ldv_loss = ldv_loss + DPP_loss_adjacency(Ax) + DPP_loss_adjacency(Az)
            torch.cuda.synchronize(); print(f'  ldv OK', flush=True)

        mi_loss_val = mi_loss_val + mi_loss(latent_codes, latent_code_guess)
        measured = torch.stack([Ax, Ay, Az], dim=-1)
        conform_loss = conform_loss + constraint_loss(measured, target_adjacency)
        torch.cuda.synchronize(); print(f'  mi+conform OK', flush=True)
    except Exception as e:
        torch.cuda.synchronize()
        print(f'  FAILED at sl={seq_length}: {type(e).__name__}: {e}', flush=True)
        raise

print(f'\nAccumulated losses OK. Now total + backward...', flush=True)
total = 20 * overlap_loss + 10 * volume_deficit_loss + 4 * compactness_loss + 0.5 * ldv_loss \
        + 1 * dpp_aspect_ratios + 2.0 * mi_loss_val + 30 * conform_loss
total.backward()
torch.cuda.synchronize(); print('backward OK')
optimizer.step()
torch.cuda.synchronize(); print('optimizer.step OK')
print('All done')
