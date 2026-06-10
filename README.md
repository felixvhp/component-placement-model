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
├── scripts/                       # runnable entry points
│   ├── train.py                   # Thin CLI (delegates to cpm.training)
│   ├── eval.py                    # Evaluation: metrics, smoothness, density sweeps
│   ├── interactive.py             # Interactive layout visualization tool
│   ├── bench_speed.py             # Inference speed benchmark
│   └── bootstrap_env.py           # Conda environment setup
├── cpm/                           # importable package (pip install -e .)
│   ├── config.py                  # Shared constants (box dims, latent dim, volume limits)
│   ├── inference.py               # Constraint-aware optimized inference
│   ├── losses.py                  # DPP / mutual-information diversity losses (train + eval)
│   ├── training/                  # Training: model factory, loss orchestration, loop
│   │   ├── factory.py             #   make_model / ablation builder
│   │   ├── objective.py           #   compute_losses
│   │   └── loop.py                #   train(args)
│   ├── constraints/               # Constraint domain
│   │   ├── sampling.py            #   target sampling & satisfaction metrics
│   │   └── projection.py          #   Levenberg–Marquardt constraint projection
│   ├── geometry/                  # Differentiable box geometry & internals
│   └── models/
│       ├── baseline.py            # Autoregressive MLP            (mlp)
│       ├── atiss.py               # ATISS-style AR transformer    (ctatiss)
│       ├── parallel_transformer.py# Parallel decoder             (parallel)
│       ├── conditional_parallel.py# Constraint-conditioned parallel (cond_parallel)
│       ├── dit_block.py           # DiT adaLN-Zero transformer blocks
│       └── components.py          # Shared building blocks (FourierEncoding, AuxiliaryPredictor)
├── slurm/                         # SLURM job scripts (training, ablations, benchmarks)
├── tests/
└── latex/main.pdf                 # Compiled report (LaTeX sources gitignored)
```

## Setup
```
python scripts/bootstrap_env.py
conda activate gcp-env
pip install -e .          # makes `import cpm` work from anywhere
```

## Usage

Train a model (see `--help` for all options):
```
python scripts/train.py --model cond_parallel --training ramp --seed 0
```

Evaluate a trained checkpoint:
```
python scripts/eval.py --weights path/to/model.pth --model cond_parallel --conditional
```

On the cluster, batched runs are launched via the scripts in `slurm/`, e.g.:
```
sbatch slurm/train_conditional.slurm
```

---

## Roadmap

- [x] Phase 1 (Weeks 1–2): Literature review
- [x] Phase 2 (Weeks 3–5): Baseline unconditional placement model
- [x] Phase 3 (Weeks 6–9): Constraint-conditioned placement model
- [x] Phase 4 (Week 10): Differentiability demonstration
- [x] Phase 5 (Weeks 11–14): Final report and code cleanup

---

## Handover / Next Steps

For anyone continuing this project:

- **Where it stands.** The four box generators (`mlp`, `ctatiss`, `parallel`, `cond_parallel`) are implemented, trained, and compared in the report. Canonical checkpoints live in `checkpoints/`, and every result maps to a job in `slurm/` (see `slurm/README.md`).
- **GLUE integration is documented, not yet wired.** `INTEGRATION.md` gives the exact contract for dropping a model into GLUE's internals subsystem, but nothing here imports `UAV` yet — actually plugging it in is the first concrete task. `cond_parallel` additionally needs GLUE's `internals_inference_batched` extended to pass a constraint `target` (see `INTEGRATION.md`).
- **Research directions** are in the report's *Limitations and Future Work* (and noted throughout) — see `latex/main.pdf`.

---

## References

Key references: Vaswani et al. (Transformers), Ho et al. (Diffusion Models), Aebersold et al. (GLUE), and layout generation works including LayoutDM, ATISS, and House-GAN.

---
