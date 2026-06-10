# Constraints

The constraint domain: how relational placement targets are sampled and scored,
and how raw layouts are projected onto the feasible set. Public names are
re-exported from the package `__init__`, so callers use
`from cpm.constraints import sample_target, make_constraint_fn, ...`.

## Files

- **`sampling.py`**: `sample_target` (signed pairwise adjacency targets at a given density/strength), `satisfaction_rate` (fraction of constraints met), and `constraint_loss` (training penalty for the conditional model)
- **`projection.py`**: `lm_project_step` (one differentiable Levenberg–Marquardt step projecting a layout toward volume/overlap feasibility), `make_constraint_fn` (builds the residual function for a batch), and `gated_ema_update` (EMA update of the projector's Lagrange multipliers)
