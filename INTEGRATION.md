# Integrating this model into GLUE

This repository is a research extension of GLUE's **internals subsystem**
(`UAV/subsystems/internals_model/` in the GLUE repo). The `mlp` baseline here
(`AutoRegressiveBoxGenerator`, `src/models/baseline.py`) *is* GLUE's current
internals generator; the other models (`ctatiss`, `parallel`, `cond_parallel`)
are the improved architectures from the report. This doc explains how to swap
one of them in.

## The contract GLUE expects

GLUE consumes the internals generator through three things
(`UAV/subsystems/forward/subsystem_inference.py`):

```python
# 1. import the class + the latent/volume constants
from UAV.subsystems.internals_model.autoregressive_generator import (
    AutoRegressiveBoxGenerator as InternalsGenerator,
    latent_dim, volume_limits,
)
# 2. import the inference routine
from UAV.subsystems.internals_model.inference_optimized import batch_optimized_inference

# 3. build, load weights, freeze
internals_generator = InternalsGenerator(device=device).to(device)
internals_generator.load_state_dict(
    torch.load(".../internals_model/box_arrangement_model.pth", map_location=device)
)
internals_generator.eval()

# 4. run (inside internals_inference_batched)
xyz, lwh = batch_optimized_inference(internals_generator, vol_req, z_internals)
```

So the module `autoregressive_generator.py` must export:

| Symbol | This repo's equivalent | Value |
|---|---|---|
| `AutoRegressiveBoxGenerator` (class) | any model in `src/models/` | `forward(volume, latent_z) -> (boxes, z_pred)` |
| `latent_dim` | `src/config.py` | `4` |
| `volume_limits` | `src/config.py` | `[0.005, 0.1]` |
| `batch_optimized_inference` | `src/inference.py` | `(model, vols, z) -> (xyz, lwh)` |

**Tensor shapes** (all `[batch, ...]`):
- `vol_req` / `volume`: `[B, n_boxes]` target box volumes
- `z_internals` / `latent_z`: `[B, 4]` — the latent GLUE controls and backprops into
- `boxes`: `[B, n_boxes, 6]` in order `(l, w, h, x, y, z)`
- `batch_optimized_inference` returns `(xyz, lwh)` = `boxes[:, :, 3:], boxes[:, :, :3]`

Every model here (`mlp`, `ctatiss`, `parallel`, `cond_parallel`) already honors
`forward(volume, latent_z)`, so the unconditional ones are drop-in compatible.

## How to swap in a new model

### Option A — install this repo as a package (recommended)

1. In GLUE's environment: `pip install -e /path/to/component-placement-model`
   (requires the packaging step — see Tier 3 in the repo's cleanup notes; the
   package would be importable as e.g. `cpm`).
2. Re-export from GLUE's `internals_model/autoregressive_generator.py`:
   ```python
   from cpm.models.parallel_transformer import ParallelBoxGenerator as AutoRegressiveBoxGenerator
   from cpm.config import latent_dim, volume_limits
   ```
   and point `inference_optimized.py` at `from cpm.inference import batch_optimized_inference`.
3. Drop the trained checkpoint in place (see below).

No changes to GLUE's calling code — `load_subsystem_models` and
`internals_inference_batched` keep working.

### Option B — copy files in-place

Copy `src/models/{parallel_transformer,dit_block,components}.py`, `src/config.py`,
and `src/inference.py` into `internals_model/`, fix the relative imports, and
alias the class to `AutoRegressiveBoxGenerator` in `autoregressive_generator.py`.
Simpler to reason about, but duplicates code and drifts from this repo.

### Checkpoint

GLUE loads `internals_model/box_arrangement_model.pth` with **strict** state-dict
loading. Provide the trained weights for your chosen model, e.g. rename
`experiments/parallel_projector_plain_s0.pth` → `box_arrangement_model.pth`. The
state-dict must match the swapped-in class (it will, since GLUE constructs the
same class it loads).

## Caveats

1. **Inference behavior changed.** This repo's `batch_optimized_inference` keeps
   the same signature as GLUE's (`(model, vols, z) -> (xyz, lwh)`, so existing
   calls work), but the defaults differ: here `method="projector"` and
   `iterations=10`; GLUE's original was the overlap gradient-descent cleanup with
   `iterations=5`. To reproduce GLUE's old cleanup, call with the non-projector
   path / matching iterations, or accept the LM-projection cleanup.

2. **`cond_parallel` needs an interface extension.** The constraint-conditioned
   model takes an extra `target` (signed pairwise adjacency) argument. GLUE's
   `internals_inference_batched(generator, z, vol)` does not pass one, so out of
   the box `cond_parallel` runs unconditioned. To use constraints in GLUE, thread
   a `target` through `internals_inference_batched` →
   `batch_optimized_inference(..., target=...)` → `model(volume, latent_z, target=...)`.

3. **Differentiability.** GLUE freezes the generator (`.eval()`, params not
   updated) but backprops into `z_internals`. All models here keep the latent
   gradient path intact, and the projector cleanup is differentiable. The report
   covers latent smoothness / gradient-flow behavior per architecture.

## Recommended model

Per the report, the attention models substantially beat the MLP baseline at
`N=10` (lower overlap, higher volume accuracy). For unconditional placement use
`parallel`; for constraint-aware placement use `cond_parallel` (with the
interface extension in caveat 2).
