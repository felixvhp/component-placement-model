#!/bin/bash
#SBATCH --job-name=cpm_ablation
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_geforce_rtx_4090:1

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: submit with --array=0-69"
    exit 1
fi

# 7 ablations × 10 seeds = 70 jobs
ABLATIONS=(z_token z_both linear_vol no_vol_concat small large no_mi)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$(( IDX % 10 ))
ABL_IDX=$(( IDX / 10 ))

ABLATION=${ABLATIONS[$ABL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda
source ~/venv/component-placement/bin/activate
cd ~/projects/component-placement-model

echo "=== Training: parallel ablation=$ABLATION seed=$SEED ==="
python -u train2.py \
    --model parallel \
    --training projector \
    --dpp plain \
    --seed $SEED \
    --max-seq-length 10 \
    --ablation $ABLATION \
    --run-name "parallel_${ABLATION}_s${SEED}"

WEIGHTS="experiments/parallel_projector_plain_${ABLATION}_s${SEED}.pth"

if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: training did not produce $WEIGHTS"
    exit 1
fi

# Eval at both seq_lens, both cleanup states
for SEQ_LEN in 3 10; do
    for CLEANUP in off on; do
        echo "=== Eval: ablation=$ABLATION sl=$SEQ_LEN cleanup=$CLEANUP ==="
        python -u eval.py \
            --model parallel \
            --ablation $ABLATION \
            --weights $WEIGHTS \
            --seq_len $SEQ_LEN \
            --num_seeds 1 \
            --cleanup $CLEANUP \
            --output_dir "results/ablation/${ABLATION}_sl${SEQ_LEN}_s${SEED}" \
            --skip_smoothness
    done
done
