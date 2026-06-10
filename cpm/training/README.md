# Training

Training infrastructure: model construction, per-batch loss orchestration, and
the training loop. The `scripts/train.py` CLI is a thin wrapper that parses
arguments and calls `train()` from here.

**For the training recipes and objective weighting, see the project report
(`../../latex/main.pdf`).**

## Files

- **`factory.py`**: `make_model` (name → model) and `make_model_with_ablation` (builds the Parallel model with the requested ablation: z-conditioning, volume encoding, head, and size variants)
- **`objective.py`**: `compute_losses` — assembles the per-batch loss dict (overlap, volume, compactness, DPP aspect-ratio / LDV, mutual information, and conditional adjacency conformity)
- **`loop.py`**: `train(args)` — builds the model, runs either the **ramp** (annealed weights) or **projector** (Levenberg–Marquardt + Lagrangian EMA) objective, logs to W&B, and saves the checkpoint to `experiments/`

## Objectives

- **`--training ramp`**: annealed loss-weight schedule (`config.get_loss_weights`); used for the MLP baseline and the conditional model
- **`--training projector`**: differentiable projection cleanup with Augmented-Lagrangian-style multiplier EMA; used for the attention models in the headline comparison
