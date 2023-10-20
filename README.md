# VQGAN-GPT for video
This is my personal copy of Songwei Ge's [TATS](https://github.com/SongweiGe/TATS) repository for video generation customized for my own purposes.

## Training
The VQGAN and GPT parts of the model are trained separately (first VQGAN, then GPT). Examples of how to train each part are demonstrated below. 

**VQGAN:** To train a VQGAN model on a set of videos: 
```python
python -u train_vqgan.py \
    --embedding_dim 256 \
    --n_codes 16384 \
    --n_hiddens 32 \
    --downsample 2 16 16 \
    --no_random_restart \
    --gpus 4 \
    --sync_batchnorm \
    --batch_size 2 \
    --num_workers 16 \
    --accumulate_grad_batches 8 \
    --progress_bar_refresh_rate 1000 \
    --max_steps 2000000 \
    --gradient_clip_val 1.0 \
    --lr 0.00005 \
    --data_path DATA_PATH \
    --default_root_dir OUTPUT_DIR \
    --resolution 256 \
    --sequence_length 32 \
    --frame_rate 16 \
    --discriminator_iter_start 13000 \
    --norm_type batch \
    --perceptual_weight 4 \
    --image_gan_weight 1 \
    --video_gan_weight 1 \
    --gan_feat_weight 4
```

Description of some of the flags:
- `data_path`: path to the dataset folder.
- `default_root_dir`: path to save the checkpoints and the tensorboard logs.
- `resolution`: the resolution of the training video clips.
- `sequence_length`: frame number of the training video clips.
- `discriminator_iter_start`: the step id to start the GAN losses.
- `downsample`: sample rate in the dimensions of time, height, and width.
- `no_random_restart`: whether to re-initialize the codebook tokens.

**GPT:** To train a GPT model on a set of videos encoded with an already trained VQGAN model: 
```python
python -u train_gpt.py \
    --gpus 4 \
    --batch_size 6 \
    --accumulate_grad_batches 4 \
    --frame_rate 8 \
    --resolution 256 \
    --unconditional \
    --vqvae VQGAN_CHECKPOINT \
    --data_path DATA_PATH \
    --default_root_dir OUTPUT_DIR \
    --vocab_size 16384 \
    --block_size 4096 \
    --n_layer 24 \
    --n_head 16 \
    --n_embd 1024 \
    --sequence_length 32 \
    --num_workers 16 \
    --val_check_interval 2 \
    --progress_bar_refresh_rate 1000 \
    --sync_batchnorm \
    --max_steps 2000000
```

Description of some of the flags:
- `vqvae`: path to the trained VQGAN checkpoint.
- `unconditional`: when no conditional information is available.

To train a conditional transformer, remove the `--unconditional` flag and use the following flags as necessary instead:
- `cond_stage_key`: what kind of conditional information to be used. It can be `label`, `text`, or `stft`.
- `stft_vqvae`: path to the trained VQGAN checkpoint for STFT features.
- `text_cond`: use this flag to indicate BPE encoded text.

## Generation
**Short videos:** To sample the videos of the same length as the training data:
```python
python -u sample_short_videos.py \
    --gpt_ckpt GPT_CHECKPOINT \
    --vqgan_ckpt VQGAN_CHECKPOINT \
    --save_name INFORMATIVE_SAVE_NAME \
    --n_sample 64 \
    --batch_size 16 \
    --resolution 256
```

Description of some of the flags:
- `gpt_ckpt`: path to the trained transformer checkpoint.
- `vqgan_ckpt`: path to the trained VQGAN checkpoint.
- `save_name`: generated samples will be saved at this location.
- `class_cond`: indicates that class labels are used as conditional information.

To compute the FVD (Frechet video distance), the following flags are required:
- `compute_fvd`: indicates that FVD will be calculated.
- `data_path`: path to the dataset folder.
- `dataset`: dataset name.
- `sample_every_n_frames`: number of frames to skip in the real video data.
- `resolution`: the resolution of real videos to compute FVD.

**Long videos:** To sample the videos longer than the training length with a sliding window:
```python
python -u sample_long_videos.py \
    --gpt_ckpt GPT_CHECKPOINT \
    --vqgan_ckpt VQGAN_CHECKPOINT \
    --save_name INFORMATIVE_SAVE_NAME \
    --n_sample 64 \
    --batch_size 16 \
    --sample_length 64 \
    --sample_resolution 16 \
    --temporal_sample_pos 1
```

Description of some of the flags:
- `sample_length`: number of latent frames to be generated.
- `temporal_sample_pos`: stride of the sliding window when generating extra frames (default: 1).

## Acknowledgments
The code is based on [VQGAN](https://github.com/CompVis/taming-transformers), [VideoGPT](https://github.com/wilson1yan/VideoGPT), and [TATS](https://github.com/SongweiGe/TATS) repositories.