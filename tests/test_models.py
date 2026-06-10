"""Smoke test: every box-generator model builds and runs a forward pass
producing correctly shaped, finite output."""

import pytest
import torch

from cpm.config import box_dim, latent_dim, volume_limits
from cpm.models.atiss import CTATISSBoxGenerator
from cpm.models.baseline import AutoRegressiveBoxGenerator
from cpm.models.conditional_parallel import ConditionalParallelBoxGenerator
from cpm.models.parallel_transformer import ParallelBoxGenerator

MODELS = {
    "mlp": AutoRegressiveBoxGenerator,
    "ctatiss": CTATISSBoxGenerator,
    "parallel": ParallelBoxGenerator,
    "cond_parallel": ConditionalParallelBoxGenerator,
}

BATCH = 4
SEQ_LEN = 5


def _sample_volumes(n, seq_len):
    lo, hi = volume_limits
    return lo + torch.rand(n, seq_len) * (hi - lo)


@pytest.mark.parametrize("name", list(MODELS))
def test_model_forward_shapes(name):
    torch.manual_seed(0)
    model = MODELS[name]().eval()

    volumes = _sample_volumes(BATCH, SEQ_LEN)
    latent_z = torch.randn(BATCH, latent_dim)

    with torch.no_grad():
        boxes, z_pred = model(volumes, latent_z)

    assert boxes.shape == (BATCH, SEQ_LEN, box_dim)
    assert torch.isfinite(boxes).all()
