#!/bin/bash
#SBATCH --job-name=cpm_train
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@ethz.ch

module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda
source ~/venv/component-placement/bin/activate

cd ~/projects/component-placement-model
python train.py
