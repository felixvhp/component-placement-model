#!/bin/bash
#SBATCH --job-name=cpm_mlp_ramp
#SBATCH --output=logs/mlp_ramp_%A_%a.out
#SBATCH --error=logs/mlp_ramp_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_geforce_rtx_4090:1

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: submit with --array=0-9"
    exit 1
fi

SEED=$SLURM_ARRAY_TASK_ID

module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda
source ~/venv/component-placement/bin/activate
cd ~/projects/component-placement-model

echo "=== Training: mlp ramp seed=$SEED ==="
python -u scripts/train.py \
    --model mlp \
    --training ramp \
    --seed $SEED \
    --max-seq-length 10 \
    --run-name "mlp_ramp_s${SEED}"

WEIGHTS="experiments/mlp_ramp_plain_s${SEED}.pth"

if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: training did not produce $WEIGHTS"
    exit 1
fi

# Eval at both seq_lens, both cleanup states
for SEQ_LEN in 3 10; do
    for CLEANUP in off on; do
        echo "=== Eval: mlp_ramp sl=$SEQ_LEN cleanup=$CLEANUP ==="
        python -u scripts/eval.py \
            --model mlp \
            --weights $WEIGHTS \
            --seq_len $SEQ_LEN \
            --num_seeds 1 \
            --cleanup $CLEANUP \
            --output_dir "results/uncond_ramp/mlp_ramp_sl${SEQ_LEN}_s${SEED}" \
            --skip_smoothness
    done
done
