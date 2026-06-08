#!/bin/bash
#SBATCH --job-name=cpm_cond_april_unif
#SBATCH --output=logs/cond_april_unif_%A_%a.out
#SBATCH --error=logs/cond_april_unif_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --array=0-9
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@ethz.ch

# 10 seeds of cond_parallel with the April 24 recipe + per-example density U(0, 0.5).
#   - loops sl=1..max_sl/epoch (matches April 24 baseline)
#   - density ~ U(0, 0.5) per example (vs. baseline's fixed 0.3)
#   - target_strength = 0.8 (the bit that the May reproduction missed)
# Submit with:  sbatch slurm/train_cond_april_unif.slurm

SEED=$SLURM_ARRAY_TASK_ID
MODEL=cond_parallel
TRAINING=ramp
RECIPE=april_unif

module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda
source ~/venv/component-placement/bin/activate
cd ~/projects/component-placement-model

# ─── Train ──────────────────────────────────────────
echo "=== Training: model=$MODEL seed=$SEED recipe=$RECIPE ==="
python -u train.py \
    --model $MODEL \
    --training $TRAINING \
    --dpp plain \
    --cond-recipe $RECIPE \
    --seed $SEED \
    --max-seq-length 10 \
    --run-name "${MODEL}_${RECIPE}_s${SEED}"

WEIGHTS="experiments/${MODEL}_${TRAINING}_plain_${RECIPE}_s${SEED}.pth"

if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: training did not produce $WEIGHTS"
    exit 1
fi

# ─── Eval ───────────────────────────────────────────
# Conditional: seq_len=10 × 2 cleanup = 2 eval calls (density sweep auto-runs inside each)
for CLEANUP in off on; do
    echo "=== Eval: $MODEL seed=$SEED cleanup=$CLEANUP ==="
    python -u eval.py \
        --model $MODEL \
        --weights $WEIGHTS \
        --seq_len 10 \
        --num_seeds 1 \
        --cleanup $CLEANUP \
        --conditional \
        --output_dir "results/cond_april_unif/${MODEL}_s${SEED}" \
        --skip_smoothness
done

echo "=== Done: model=$MODEL seed=$SEED recipe=$RECIPE ==="
