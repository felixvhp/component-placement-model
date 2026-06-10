# cpm — Component Placement Model

Importable package for the generative 3D component-placement model. It produces
constraint-respecting, axis-aligned box layouts in a fully differentiable manner,
and is designed to drop into GLUE's internals subsystem (see `../INTEGRATION.md`).

**For the model architectures, losses, and evaluation methodology, see the
project report (`../latex/main.pdf`).**

## Files

- **`config.py`**: Shared constants (box/latent dimensions, volume limits) plus the ramp loss-weight schedule, projector configuration, and projector objective weights
- **`inference.py`**: `batch_optimized_inference` — constraint-aware inference with the differentiable cleanup/projection step; the entry point GLUE calls
- **`losses.py`**: Diversity losses used both as training objectives and evaluation metrics (DPP variants for LDV, mutual information)

## Subpackages

- **`models/`**: The box-generator architectures (`mlp`, `ctatiss`, `parallel`, `cond_parallel`) and their shared building blocks
- **`training/`**: Model factory, per-batch loss orchestration, and the training loop
- **`constraints/`**: Constraint target sampling, satisfaction metrics, and Levenberg–Marquardt projection
- **`geometry/`**: Differentiable box geometry (overlap, volume, adjacency); adapted from GLUE's `UAV/geometry/`
