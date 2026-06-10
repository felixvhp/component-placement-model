#!/bin/bash
#SBATCH --job-name=cpm_conditional
#SBATCH --output=logs/conditional_%A_%a.out
#SBATCH --error=logs/conditional_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --array=0-9
#SBATCH --exclude=eu-lo-g3-059
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@ethz.ch

# 10 seeds of cond_parallel (the conditional model used in the report):
#   - loops sl=1..max_sl/epoch
#   - constraint density = 0.3 fixed
#   - target_strength = 0.8
#   - slenderness + bbox_volume compactness
# Submit with:  sbatch slurm/train_conditional.slurm

SEED=$SLURM_ARRAY_TASK_ID
MODEL=cond_parallel
TRAINING=ramp

module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda
source ~/venv/component-placement/bin/activate
cd ~/projects/component-placement-model

# ─── Train ──────────────────────────────────────────
echo "=== Training: model=$MODEL seed=$SEED ==="
python -u scripts/train.py \
    --model $MODEL \
    --training $TRAINING \
    --seed $SEED \
    --max-seq-length 10 \
    --run-name "${MODEL}_s${SEED}"

WEIGHTS="experiments/${MODEL}_${TRAINING}_plain_s${SEED}.pth"

if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: training did not produce $WEIGHTS"
    exit 1
fi

# ─── Eval ───────────────────────────────────────────
# Conditional: seq_len=10 and seq_len=3, both cleanup on/off (density sweep auto-runs).
for SEQ_LEN in 3 10; do
    for CLEANUP in off on; do
        echo "=== Eval: $MODEL seed=$SEED sl=$SEQ_LEN cleanup=$CLEANUP ==="
        python -u scripts/eval.py \
            --model $MODEL \
            --weights $WEIGHTS \
            --seq_len $SEQ_LEN \
            --num_seeds 1 \
            --cleanup $CLEANUP \
            --conditional \
            --output_dir "results/conditional/${MODEL}_s${SEED}" \
            --skip_smoothness
    done
done

echo "=== Done: model=$MODEL seed=$SEED ==="
