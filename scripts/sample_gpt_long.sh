#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=00:40:00
#SBATCH --job-name=sample_gpt_long
#SBATCH --output=sample_gpt_long_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

srun python -u ../sample_long_videos.py \
    --gpt_ckpt /scratch/eo41/vqgan-gpt-video/models_gpt/vqgan_256_8x16x16_8fps_28k_gpt_40k.ckpt \
    --vqgan_ckpt /scratch/eo41/vqgan-gpt-video/models_vqgan/vqgan_256_8x16x16_8fps_28k.ckpt \
    --save_name vqgan_256_8x16x16_8fps_28k_gpt_40k \
    --n_sample 64 \
    --batch_size 16 \
    --sample_length 64 \
    --sample_resolution 16 \
    --temporal_sample_pos 1

echo "Done"