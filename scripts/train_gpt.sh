#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480GB
#SBATCH --time=48:00:00
#SBATCH --job-name=train_gpt
#SBATCH --output=train_gpt_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=4

srun python -u ../train_gpt.py \
    --gpus 4 \
    --batch_size 6 \
    --accumulate_grad_batches 4 \
    --frame_rate 8 \
    --resolution 256 \
    --unconditional \
    --vqvae "/scratch/eo41/vqgan-gpt-video/models_vqgan/vqgan_256_8x16x16_8fps_28k.ckpt" \
    --data_path "/scratch/eo41/data-video/minute/S" \
    --default_root_dir "/scratch/eo41/vqgan-gpt-video/models_gpt/model_256_8x16x16_8fps_28k" \
    --vocab_size 16384 \
    --block_size 2048 \
    --n_layer 24 \
    --n_head 16 \
    --n_embd 1024 \
    --sequence_length 16 \
    --num_workers 16 \
    --val_check_interval 0.5 \
    --progress_bar_refresh_rate 1000 \
    --sync_batchnorm \
    --max_steps 2000000

echo "Done"