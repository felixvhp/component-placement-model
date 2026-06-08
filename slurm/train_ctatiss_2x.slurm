#!/bin/bash
#SBATCH --job-name=cpm_ctatiss_2x
#SBATCH --output=logs/ctatiss_2x_%j.out
#SBATCH --error=logs/ctatiss_2x_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --exclude=eu-lo-g3-059

# One-off: ctatiss with projector training + 2x ldv and 2x infonce loss weights.
# Under --training projector, the relevant weights live in config.objective_weights:
#   ldv: 0.5 -> 1.0    infonce: 0.1 -> 0.2

SEED=0
MODEL=ctatiss
TRAINING=projector

module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda
source ~/venv/component-placement/bin/activate
cd ~/projects/component-placement-model

echo "=== Training: $MODEL seed=$SEED training=$TRAINING ldv=2x infonce=2x ==="
python -u train.py \
    --model $MODEL \
    --training $TRAINING \
    --dpp plain \
    --ldv-scale 2.0 \
    --infonce-scale 2.0 \
    --seed $SEED \
    --max-seq-length 10 \
    --run-name "${MODEL}_${TRAINING}_ldv2x_infonce2x_s${SEED}"

WEIGHTS="experiments/${MODEL}_${TRAINING}_plain_ldv2x_infonce2x_s${SEED}.pth"

if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: training did not produce $WEIGHTS"
    exit 1
fi

# Eval: sl=3 and sl=10, cleanup off/on
for SEQ_LEN in 3 10; do
    for CLEANUP in off on; do
        echo "=== Eval: sl=$SEQ_LEN cleanup=$CLEANUP ==="
        python -u eval.py \
            --model $MODEL \
            --weights $WEIGHTS \
            --seq_len $SEQ_LEN \
            --num_seeds 1 \
            --cleanup $CLEANUP \
            --output_dir "results/ctatiss_2x_s${SEED}" \
            --skip_smoothness
    done
done

echo "=== Done ==="
