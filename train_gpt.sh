#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480GB
#SBATCH --time=00:10:00
#SBATCH --job-name=train_gpt
#SBATCH --output=train_gpt_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=4

srun python -u train_gpt.py \
    --num_workers 16 \
    --val_check_interval 0.5 \
    --progress_bar_refresh_rate 500 \
    --gpus 4 \
    --sync_batchnorm \
    --batch_size 8 \
    --unconditional \
    --vqvae "/scratch/eo41/vqgan-gpt-video/models_vqgan/vqgan.ckpt" \
    --data_path "/scratch/eo41/data-video/minute/S" \
    --default_root_dir "/scratch/eo41/vqgan-gpt-video/models_gpt" \
    --vocab_size 16384 \
    --block_size 1024 \
    --n_layer 24 \
    --n_head 16 \
    --n_embd 1024 \
    --resolution 128 \
    --sequence_length 16 \
    --max_steps 2000000

echo "Done"