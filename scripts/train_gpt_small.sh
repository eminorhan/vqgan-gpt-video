#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480GB
#SBATCH --time=48:00:00
#SBATCH --job-name=train_gpt_small
#SBATCH --output=train_gpt_small_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=4

# s
srun python -u ../train_gpt.py \
    --gpus 4 \
    --batch_size 1 \
    --accumulate_grad_batches 32 \
    --base_lr 0.000002 \
    --frame_rate 8 \
    --resolution 192 \
    --unconditional \
    --vqvae "/scratch/eo41/vqgan-gpt-video/models_vqgan/vqgan_192_8x24x24_8fps_57k.ckpt" \
    --data_path "/scratch/eo41/data-video/minute/S" \
    --default_root_dir "/scratch/eo41/vqgan-gpt-video/models_gpt/model_192_8x24x24_8fps_57k_small" \
    --vocab_size 16384 \
    --block_size 4608 \
    --n_layer 24 \
    --n_head 16 \
    --n_embd 1024 \
    --sequence_length 16 \
    --num_workers 16 \
    --check_val_every_n_epoch 100 \
    --progress_bar_refresh_rate 1000 \
    --sync_batchnorm \
    --max_steps 2000000

# # kinetics
# srun python -u ../train_gpt.py \
#     --gpus 4 \
#     --batch_size 1 \
#     --accumulate_grad_batches 32 \
#     --base_lr 0.000001 \
#     --frame_rate 16 \
#     --resolution 256 \
#     --unconditional \
#     --vqvae "/scratch/eo41/vqgan-gpt-video/models_vqgan/vqgan_256_16x16x16_16fps_kinetics_13k.ckpt" \
#     --data_path "/scratch/eo41/data-video/minute/S" \
#     --default_root_dir "/scratch/eo41/vqgan-gpt-video/models_gpt/model_256_16x16x16_16fps_kinetics_13k" \
#     --vocab_size 16384 \
#     --block_size 4096 \
#     --n_layer 32 \
#     --n_head 20 \
#     --n_embd 1280 \
#     --sequence_length 32 \
#     --num_workers 16 \
#     --check_val_every_n_epoch 100 \
#     --progress_bar_refresh_rate 1000 \
#     --sync_batchnorm \
#     --max_steps 2000000

echo "Done"