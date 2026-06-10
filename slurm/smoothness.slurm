#!/bin/bash
#SBATCH --job-name=cpm_smooth
#SBATCH --output=logs/smoothness_%j.out
#SBATCH --error=logs/smoothness_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_geforce_rtx_4090:1

module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda
source ~/venv/component-placement/bin/activate
cd ~/projects/component-placement-model

# Generate the 6 latent-smoothness figures for the report.
# Uses the seed-0 projector checkpoints (same models as the report).
# model_key : checkpoint : latex_prefix
declare -A CKPT=(
    [mlp]="experiments/mlp_projector_plain_s0.pth"
    [ctatiss]="experiments/ctatiss_projector_plain_s0.pth"
    [parallel]="experiments/parallel_projector_plain_s0.pth"
)
declare -A PREFIX=( [mlp]="mlp_" [ctatiss]="ct_" [parallel]="p_" )

mkdir -p latex/smoothness

for MODEL in mlp ctatiss parallel; do
    for CLEANUP in off on; do
        OUT="plots/smoothness/${MODEL}_cleanup-${CLEANUP}"
        mkdir -p "$OUT"
        echo "=== smoothness: $MODEL cleanup=$CLEANUP ==="
        python -u scripts/eval.py \
            --model "$MODEL" \
            --weights "${CKPT[$MODEL]}" \
            --seq_len 10 \
            --num_seeds 1 \
            --cleanup "$CLEANUP" \
            --output_dir "$OUT" \
            --skip_density_sweep
        # copy the x-position plot to latex with the report's prefix
        cp "$OUT/smoothness_x_sl10_cleanup-${CLEANUP}.png" \
           "latex/smoothness/${PREFIX[$MODEL]}smoothness_x_sl10_cleanup-${CLEANUP}.png"
    done
done

echo "=== done; updated files: ==="
ls -la latex/smoothness/*smoothness_x_sl10*.png
