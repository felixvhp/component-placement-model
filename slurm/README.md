# SLURM Jobs

Batch job scripts for the Euler cluster. Submit from the repository root, e.g.
`sbatch slurm/train_cond_april.slurm`. Each script loads the environment,
`cd`s to the project root, and runs training and/or evaluation. Most are arrays
over the 10 training seeds.

## Files

- **`train.slurm`**: Combined run (40-job array) — unconditional models (`mlp`, `ctatiss`, `parallel`) with the projector objective, plus `cond_parallel`; produces the headline comparison
- **`mlp_ramp.slurm`**: MLP baseline with the ramp schedule → the baseline table
- **`parallel_ablations.slurm`**: Parallel decoder ablations (Base, A–F) → the ablation table
- **`train_cond_april.slurm`**: `cond_parallel` with the april recipe (density 0.3, target_strength 0.8) → the conditional table, plus its evaluation
- **`retrain_ctatiss_s7.slurm`**: Retrain a specific `ctatiss` seed
- **`regen_smoothness.slurm`**: Re-run evaluation to regenerate latent-smoothness plots
- **`bench_speed.slurm`**: Run the inference speed benchmark → `results/speed_benchmark.json`

> Submit the combined `train.slurm` with `--array=0-39`.
