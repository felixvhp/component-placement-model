# Geometry

Differentiable geometry layer for axis-aligned component boxes: overlap/collision
checking, volume, rough view areas, and pairwise adjacency. Adapted from GLUE's
`UAV/geometry/` so the internals subsystem stays gradient-compatible.

## Files

- **`box.py`**: Bounding-box primitives (`Box`, `BatchBox`) — corners, sizes, and volume
- **`internals.py`**: `Internals` / `BatchInternals` — arranges boxes from `(xyz, lwh)` and computes overlap, box volumes, rough view areas/volume, and adjacency matrices used by the losses and metrics
- **`unique_with_grad.py`**: `UniqueWithGrad`, a custom autograd function for a gradient-preserving `unique` operation

Each class has a NumPy single-instance variant and a PyTorch batched variant
(`Box`/`BatchBox`, `Internals`/`BatchInternals`); the batched ones are used in
training thanks to autograd.
