#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=300GB
#SBATCH --time=2:00:00
#SBATCH --job-name=train_vqgan_256_16x16x16_16fps_2gpu
#SBATCH --output=train_vqgan_256_16x16x16_16fps_2gpu_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=2

srun python -u ../train_vqgan.py \
    --embedding_dim 256 \
    --n_codes 16384 \
    --n_hiddens 32 \
    --downsample 2 16 16 \
    --no_random_restart \
    --gpus 2 \
    --sync_batchnorm \
    --batch_size 3 \
    --num_workers 16 \
    --accumulate_grad_batches 8 \
    --progress_bar_refresh_rate 1000 \
    --max_steps 2000000 \
    --gradient_clip_val 1.0 \
    --lr 0.00005 \
    --data_path "/scratch/eo41/data-video/minute/S" \
    --default_root_dir "/scratch/eo41/vqgan-gpt-video/models_256_16x16x16_16fps_2gpu" \
    --resolution 256 \
    --sequence_length 32 \
    --frame_rate 16 \
    --discriminator_iter_start 13000 \
    --norm_type batch \
    --perceptual_weight 4 \
    --image_gan_weight 1 \
    --video_gan_weight 1 \
    --gan_feat_weight 4

echo "Done"