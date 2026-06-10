#!/bin/bash
#SBATCH --job-name=cpm_train
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@ethz.ch

# 40 jobs total:
# 0-29:  unconditional (mlp, ctatiss, parallel × 10 seeds), projector
# 30-39: conditional (cond_parallel × 10 seeds), ramp

SEEDS=(0 1 2 3 4 5 6 7 8 9)
UNCOND_MODELS=(mlp ctatiss parallel)
COND_MODELS=(cond_parallel)

IDX=$SLURM_ARRAY_TASK_ID

module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda
source ~/venv/component-placement/bin/activate
cd ~/projects/component-placement-model

if [ $IDX -lt 30 ]; then
    SEED_IDX=$(( IDX % 10 ))
    MODEL_IDX=$(( IDX / 10 ))
    MODEL=${UNCOND_MODELS[$MODEL_IDX]}
    SEED=${SEEDS[$SEED_IDX]}
    TRAINING="projector"
    IS_COND=0
else
    COND_IDX=$(( IDX - 30 ))
    SEED_IDX=$(( COND_IDX % 10 ))
    MODEL_IDX=$(( COND_IDX / 10 ))
    MODEL=${COND_MODELS[$MODEL_IDX]}
    SEED=${SEEDS[$SEED_IDX]}
    TRAINING="ramp"
    IS_COND=1
fi

# ─── Train ──────────────────────────────────────────
echo "=== Training: model=$MODEL seed=$SEED training=$TRAINING ==="
python -u train.py \
    --model $MODEL \
    --training $TRAINING \
    --seed $SEED \
    --max-seq-length 10 \
    --run-name "${MODEL}_${TRAINING}_s${SEED}"

WEIGHTS="experiments/${MODEL}_${TRAINING}_plain_s${SEED}.pth"

if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: training did not produce $WEIGHTS"
    exit 1
fi

# ─── Eval ───────────────────────────────────────────
if [ $IS_COND -eq 0 ]; then
    # Unconditional: 2 seq_lens × 2 cleanup = 4 eval calls
    for SEQ_LEN in 3 10; do
        for CLEANUP in off on; do
            echo "=== Eval: $MODEL sl=$SEQ_LEN cleanup=$CLEANUP ==="
            python -u eval.py \
                --model $MODEL \
                --weights $WEIGHTS \
                --seq_len $SEQ_LEN \
                --num_seeds 1 \
                --cleanup $CLEANUP \
                --output_dir "results/uncond/${MODEL}_sl${SEQ_LEN}_s${SEED}" \
                --skip_smoothness
        done
    done
else
    # Conditional: seq_len=10 only × 2 cleanup = 2 eval calls (density sweep auto-runs)
    for CLEANUP in off on; do
        echo "=== Eval: $MODEL cleanup=$CLEANUP ==="
        python -u eval.py \
            --model $MODEL \
            --weights $WEIGHTS \
            --seq_len 10 \
            --num_seeds 1 \
            --cleanup $CLEANUP \
            --conditional \
            --output_dir "results/cond/${MODEL}_s${SEED}" \
            --skip_smoothness
    done
fi

echo "=== Done: model=$MODEL seed=$SEED ==="
