#!/bin/bash
#SBATCH --job-name=cpm_ctatiss_s7
#SBATCH --output=logs/retrain_ctatiss_s7_%j.out
#SBATCH --error=logs/retrain_ctatiss_s7_%j.err
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --exclude=eu-lo-g3-059

# Retrain ctatiss seed 7 (previous checkpoint had all-NaN weights — training diverged).
# CUDA non-determinism + the hardened _log_det fallback should let this land cleanly.

SEED=7
MODEL=ctatiss
TRAINING=projector

module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda
source ~/venv/component-placement/bin/activate
cd ~/projects/component-placement-model

# Clear stale NaN eval artifacts from the previous corrupt checkpoint
rm -f results2/uncond/ctatiss_sl3_s7/results_ctatiss_sl3_cleanup-*.json
rm -f results2/uncond/ctatiss_sl10_s7/results_ctatiss_sl10_cleanup-*.json

echo "=== Training: $MODEL seed=$SEED training=$TRAINING ==="
python -u train.py \
    --model $MODEL \
    --training $TRAINING \
    --dpp plain \
    --seed $SEED \
    --max-seq-length 10 \
    --run-name "${MODEL}_${TRAINING}_s${SEED}_retrain"

WEIGHTS="experiments/${MODEL}_${TRAINING}_plain_s${SEED}.pth"
if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: training did not produce $WEIGHTS"; exit 1
fi

# Reject a NaN-corrupt checkpoint before wasting eval time
python -c "
import torch, sys
sd = torch.load('$WEIGHTS', map_location='cpu')
n_nan = sum(int(torch.isnan(v).any()) for v in sd.values() if torch.is_tensor(v))
if n_nan:
    print(f'ERROR: checkpoint has {n_nan} NaN tensors — training diverged again')
    sys.exit(2)
print('checkpoint is clean')
" || exit 2

for SEQ_LEN in 3 10; do
    for CLEANUP in off on; do
        echo "=== Eval: sl=$SEQ_LEN cleanup=$CLEANUP ==="
        python -u eval.py \
            --model $MODEL \
            --weights $WEIGHTS \
            --seq_len $SEQ_LEN \
            --num_seeds 1 \
            --cleanup $CLEANUP \
            --output_dir "results2/uncond/${MODEL}_sl${SEQ_LEN}_s${SEED}" \
            --skip_smoothness
    done
done

echo "=== Done ==="
