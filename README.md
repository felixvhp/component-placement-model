# Generative Component Placement Model

**Semester Project — ETH Zürich, D-MAVT**
**IDEAL Lab | Supervisor: Prof. Mark Fuge | Co-Supervisor: Tim Aebersold**
**Duration: 16.02.2026 – 29.05.2026**

---

## Overview

This project develops a generative model for 3D component placement that produces constraint-respecting box configurations in a fully differentiable manner. The model is designed to integrate with the GLUE framework, enabling gradient-based coordination with system-level objectives.

---

## Research Question

How can a generative model produce constraint-respecting 3D component layouts in a fully differentiable manner, enabling gradient-based coordination with system-level objectives?

---

## Project Structure
```
component-placement-model/
├── train.py                       # Training entry point (mlp, ctatiss, parallel, cond_parallel)
├── eval.py                        # Evaluation: metrics, smoothness, density sweeps
├── interactive.py                 # Interactive layout visualization tool
├── bench_speed.py                 # Inference speed benchmark
├── bootstrap_env.py               # Conda environment setup
├── src/
│   ├── config.py                  # Shared constants (box dims, latent dim, volume limits)
│   ├── constraints.py             # Constraint sampling & satisfaction metrics
│   ├── inference.py               # Constraint-aware optimized inference
│   ├── projector.py               # Levenberg–Marquardt constraint projection
│   ├── losses.py                  # DPP / mutual-information diversity losses
│   ├── gnn_encoder.py             # GNN constraint encoder
│   ├── training_plots.py          # Training visualization helpers
│   ├── geometry/                  # Differentiable box geometry & internals
│   └── models/
│       ├── baseline.py            # Autoregressive MLP            (mlp)
│       ├── atiss.py               # ATISS-style AR transformer    (ctatiss)
│       ├── parallel_transformer.py# Parallel decoder             (parallel)
│       ├── conditional_parallel.py# Constraint-conditioned parallel (cond_parallel)
│       ├── dit_block.py           # DiT adaLN-Zero transformer blocks
│       └── components.py          # Shared building blocks (FourierEncoding)
├── slurm/                         # SLURM job scripts (training, ablations, benchmarks)
├── tests/
└── latex/                         # Report sources
```

## Setup
```
python bootstrap_env.py
conda activate gcp-env
```

## Usage

Train a model (see `--help` for all options):
```
python train.py --model cond_parallel --training ramp --cond-recipe april --seed 0
```

Evaluate a trained checkpoint:
```
python eval.py --weights path/to/model.pth --model cond_parallel --conditional
```

On the cluster, batched runs are launched via the scripts in `slurm/`, e.g.:
```
sbatch slurm/train_cond_april.slurm
```

---

## Roadmap

- [x] Phase 1 (Weeks 1–2): Literature review
- [x] Phase 2 (Weeks 3–5): Baseline unconditional placement model
- [x] Phase 3 (Weeks 6–9): Constraint-conditioned placement model
- [x] Phase 4 (Week 10): Differentiability demonstration
- [x] Phase 5 (Weeks 11–14): Final report and code cleanup

---

## References

Key references: Vaswani et al. (Transformers), Ho et al. (Diffusion Models), Aebersold et al. (GLUE), and layout generation works including LayoutDM, ATISS, and House-GAN.

---
