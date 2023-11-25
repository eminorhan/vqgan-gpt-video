#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=04:00:00
#SBATCH --job-name=sample_gpt_long
#SBATCH --output=sample_gpt_long_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

srun python -u ../sample_long_videos.py \
    --gpt_ckpt /scratch/eo41/vqgan-gpt-video/models_gpt/vqgan_192_8x24x24_16fps_54k_gpt_6k.ckpt \
    --vqgan_ckpt /scratch/eo41/vqgan-gpt-video/models_vqgan/vqgan_192_8x24x24_16fps_54k.ckpt \
    --save_name vqgan_192_8x24x24_16fps_54k_gpt_6k_uncond \
    --n_sample 64 \
    --batch_size 16 \
    --sample_length 32 \
    --sample_resolution 24 \
    --temporal_sample_pos 1

echo "Done"