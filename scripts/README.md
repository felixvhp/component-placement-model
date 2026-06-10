# Scripts

Runnable entry points. Run them from the repository root (e.g.
`python scripts/train.py ...`); imports resolve via the installed `cpm` package
(`pip install -e .`), and outputs go to `experiments/` and `results/` relative
to the working directory.

## Files

- **`train.py`**: Training CLI — parses arguments and delegates to `cpm.training.train`. Selects model (`mlp`/`ctatiss`/`parallel`/`cond_parallel`), objective (`ramp`/`projector`), and Parallel ablation
- **`eval.py`**: Evaluation — computes metrics (overlap, volume accuracy, compactness, LDV, MI, constraint satisfaction), latent smoothness plots, and density sweeps; writes JSON to `results/`
- **`interactive.py`**: Interactive matplotlib viewer with sliders for volumes, latent code, and relational constraints
- **`bench_speed.py`**: Inference speed/throughput benchmark (forward + backward, the GLUE-relevant gradient path)
- **`bootstrap_env.py`**: Cross-platform conda environment setup (creates `gcp-env`, installs the package editable, sets up pre-commit). Run from the project root
