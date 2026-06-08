#!/bin/bash
#SBATCH --job-name=cpm_bench
#SBATCH --output=logs/bench_speed_%j.out
#SBATCH --error=logs/bench_speed_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_geforce_rtx_4090:1

module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda
source ~/venv/component-placement/bin/activate
cd ~/projects/component-placement-model

echo "=== Speed benchmark: AR-Transformer vs Parallel ==="
python -u bench_speed.py \
    --batch 100 \
    --seq_lens 3 10 \
    --seeds 10 \
    --reps 50 \
    --warmup 10 \
    --out results/speed_benchmark.json
