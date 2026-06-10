# Tests

Fast smoke tests (`pytest tests/`). They guard against import/shape regressions
and confirm the training loop runs end-to-end — the kind of breakage that
structural refactors can introduce.

## Files

- **`test_models.py`**: Builds each model (`mlp`, `ctatiss`, `parallel`, `cond_parallel`) and runs a forward pass, asserting the output shape `(batch, seq_len, 6)` and finiteness
- **`test_training.py`**: Runs one epoch of both the projector and ramp objectives (W&B disabled, into a temp dir) and asserts a checkpoint is written
