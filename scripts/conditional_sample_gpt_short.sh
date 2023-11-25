#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=01:00:00
#SBATCH --job-name=conditional_sample_gpt_short
#SBATCH --output=conditional_sample_gpt_short_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

# # s
# srun python -u ../conditional_sample_short_videos.py \
#     --gpt_ckpt /scratch/eo41/vqgan-gpt-video/models_gpt/vqgan_256_8x16x16_8fps_28k_gpt_40k.ckpt \
#     --vqgan_ckpt /scratch/eo41/vqgan-gpt-video/models_vqgan/vqgan_256_8x16x16_8fps_28k.ckpt \
#     --save_name vqgan_256_8x16x16_8fps_28k_gpt_40k_cond_ctxlen_6 \
#     --batch_size 16 \
#     --num_workers 16 \
#     --resolution 256 \
#     --frame_rate 8 \
#     --sequence_length 16 \
#     --ctx_video_dir /scratch/eo41/data-video/minute/S \
#     --ctx_len 6

# # ssv2 - 50 shot
# srun python -u ../conditional_sample_short_videos.py \
#     --gpt_ckpt /scratch/eo41/vqgan-gpt-video/models_gpt/vqgan_192_8x24x24_16fps_54k_gpt_9k.ckpt \
#     --vqgan_ckpt /scratch/eo41/vqgan-gpt-video/models_vqgan/vqgan_192_8x24x24_16fps_54k.ckpt \
#     --save_name vqgan_192_8x24x24_16fps_54k_gpt_9k_ctx4 \
#     --batch_size 1 \
#     --num_workers 16 \
#     --resolution 192 \
#     --frame_rate 16 \
#     --sequence_length 16 \
#     --ctx_video_dir /vast/eo41/ssv2/train_50shot \
#     --ctx_len 4

# s 
srun python -u ../conditional_sample_short_videos.py \
    --gpt_ckpt /scratch/eo41/vqgan-gpt-video/models_gpt/vqgan_192_8x24x24_8fps_57k_gpt_5k.ckpt \
    --vqgan_ckpt /scratch/eo41/vqgan-gpt-video/models_vqgan/vqgan_192_8x24x24_8fps_57k.ckpt \
    --save_name vqgan_192_8x24x24_8fps_57k_gpt_5k_ctx2 \
    --batch_size 1 \
    --num_workers 16 \
    --resolution 192 \
    --frame_rate 8 \
    --sequence_length 16 \
    --ctx_video_dir /scratch/eo41/data-video/minute/S \
    --ctx_len 2

echo "Done"