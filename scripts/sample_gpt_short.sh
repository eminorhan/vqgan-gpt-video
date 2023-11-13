#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=02:00:00
#SBATCH --job-name=sample_gpt_short
#SBATCH --output=sample_gpt_short_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

srun python -u ../sample_short_videos.py \
    --gpt_ckpt /scratch/eo41/vqgan-gpt-video/models_gpt/vqgan_256_16x16x16_16fps_11k_gpt_9k.ckpt \
    --vqgan_ckpt /scratch/eo41/vqgan-gpt-video/models_vqgan/vqgan_256_16x16x16_16fps_11k.ckpt \
    --save_name vqgan_256_16x16x16_16fps_11k_gpt_9k_uncond \
    --n_sample 1024 \
    --batch_size 1 \
    --resolution 256

echo "Done"