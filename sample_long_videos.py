# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import tqdm
import torch
import argparse
import numpy as np
from einops import repeat

from tats import load_transformer
from tats.utils import save_video_grid
from tats.modules.gpt import sample_with_past

parser = argparse.ArgumentParser()
parser.add_argument('--gpt_ckpt', type=str, default='')
parser.add_argument('--vqgan_ckpt', type=str, default='')
parser.add_argument('--save_dir', type=str, default='../samples_long')
parser.add_argument('--save_name', type=str, default='demo')
parser.add_argument('--top_k', type=int, default=2048)
parser.add_argument('--top_p', type=float, default=0.92)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_sample', type=int, default=100)
parser.add_argument('--sample_length', type=int, default=256)
parser.add_argument('--sample_resolution', type=int, default=16)
parser.add_argument('--temporal_sample_pos', type=int, default=1)
parser.add_argument('--save_array', action='store_true')
args = parser.parse_args()


@torch.no_grad()
def sample_long(gpt, temporal_infer, spatial_infer, temporal_train, spatial_train, temporal_sample_pos, batch_size, class_label, temperature=1.):
    
    spatial_sample_pos = spatial_train // 2

    with torch.no_grad():
        c_indices = repeat(torch.tensor([class_label]), '1 -> b 1', b=batch_size).to(gpt.device)  # class token
        index_sample_all = torch.zeros([batch_size, temporal_infer, spatial_infer, spatial_infer]).long().cuda()

        for t in tqdm.tqdm(range(temporal_infer)):
            for i in range(spatial_infer):
                for j in range(spatial_infer):
                    if t <= temporal_sample_pos:
                        local_t = t
                    elif temporal_infer-t <= temporal_train-temporal_sample_pos:
                        local_t = temporal_train-(temporal_infer-t)
                    else:
                        local_t = temporal_sample_pos
                    if i <= spatial_sample_pos:
                        local_i = i
                    elif spatial_infer-i <= spatial_train-spatial_sample_pos:
                        local_i = spatial_train-(spatial_infer-i)
                    else:
                        local_i = spatial_sample_pos
                    if j <= spatial_sample_pos:
                        local_j = j
                    elif spatial_infer-j <= spatial_train-spatial_sample_pos:
                        local_j = spatial_train-(spatial_infer-j)
                    else:
                        local_j = spatial_sample_pos
                    t_start = 0
                    t_end = t
                    t_start = t-local_t
                    t_end = t_start+temporal_train
                    i_start = i-local_i
                    i_end = i_start+spatial_train
                    j_start = j-local_j
                    j_end = j_start+spatial_train
                    patch = torch.cat([c_indices, index_sample_all[:, t_start:t_end, i_start:i_end, j_start:j_end].reshape(c_indices.shape[0], -1)], 1)
                    if t_start == 0:
                        logits, _ = gpt.transformer(patch[:,:-1], cbox=[[i_start, i_end, j_start, j_end]], tbox=[[t_start, t_end]])
                    else:
                        logits, _ = gpt.transformer(patch[:,:-1], cbox=[[i_start, i_end, j_start, j_end]])
                    logits = logits.reshape(logits.shape[0], temporal_train, spatial_train, spatial_train, -1)
                    logits = logits[:, local_t, local_i, local_j, :]
                    logits = logits/temperature
                    if args.top_k is not None:
                      logits = gpt.top_k_logits(logits, args.top_k)
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    index_sample_all[:, t, i, j] = torch.multinomial(probs, num_samples=1).squeeze()

        torch.cuda.empty_cache()
        index_sample_all = torch.clamp(index_sample_all-gpt.cond_stage_vocab_size, min=0, max=gpt.first_stage_model.n_codes-1)

        x_sample = []
        for i in range(batch_size):
            x_sample.append(gpt.first_stage_model.decode(index_sample_all[i:i+1, :, :, :].cuda()).cpu())
        x_sample = torch.cat(x_sample, 0)
        x_sample = torch.clamp(x_sample, -0.5, 0.5) + 0.5
        
    return x_sample


@torch.no_grad()
def sample_long_fast(gpt, temporal_infer, spatial_infer, temporal_sample_pos, batch_size, class_label, temperature=1.):
    
    steps = slice_n_code = spatial_infer**2

    with torch.no_grad():
        index_sample_all = torch.zeros([batch_size, temporal_infer*spatial_infer*spatial_infer]).long().cuda()
        c_indices = repeat(torch.tensor([class_label]), '1 -> b 1', b=batch_size).to(gpt.device)  # class token
        index_sample_all[:,:temporal_sample_pos*steps] = sample_with_past(c_indices, gpt.transformer, steps=temporal_sample_pos*steps, sample_logits=True, top_k=args.top_k, temperature=temperature, top_p=args.top_p)
        
        for t_id in range(temporal_infer-temporal_sample_pos):
            i_start = t_id*slice_n_code
            i_end = (temporal_sample_pos+t_id)*slice_n_code
            x_past = index_sample_all[:,i_start:i_end]
            index_sample_all[:,i_end:i_end+steps] = sample_with_past(torch.cat([c_indices, x_past], dim=1), gpt.transformer, steps=steps, sample_logits=True, top_k=args.top_k, temperature=temperature, top_p=args.top_p)
        
        torch.cuda.empty_cache()
        index_sample_all = index_sample_all.reshape([batch_size, temporal_infer, spatial_infer, spatial_infer])
        index_sample_all = torch.clamp(index_sample_all-gpt.cond_stage_vocab_size, min=0, max=gpt.first_stage_model.n_codes-1)
    
        x_sample = []
        for i in range(batch_size):
            x_sample.append(gpt.first_stage_model.decode(index_sample_all[i:i+1, :, :, :].cuda()).cpu())
        x_sample = torch.cat(x_sample, 0)
        x_sample = torch.clamp(x_sample, -0.5, 0.5) + 0.5

    return x_sample


if __name__ == "__main__":

    # load pretrained GPT-video model
    gpt = load_transformer(args.gpt_ckpt, vqgan_ckpt=args.vqgan_ckpt).cuda().eval()

    temporal_train, spatial_train, _ = gpt.first_stage_model.latent_shape
    all_data = []
    n_row = int(np.sqrt(args.batch_size))

    save_path = os.path.join(args.save_dir, args.save_name)
    print(f'generating and saving video to {save_path}')
    os.makedirs(save_path, exist_ok=True)

    n_batch = args.n_sample // args.batch_size
    with torch.no_grad():
        for sample_id in tqdm.tqdm(range(n_batch)):
            x_sample = sample_long_fast(gpt, args.sample_length, args.sample_resolution, temporal_sample_pos=args.temporal_sample_pos, batch_size=args.batch_size, class_label=0)
            print(f'generated sample {sample_id}')
            save_video_grid(x_sample, os.path.join(save_path, f'sample_{sample_id}.mp4'), n_row)

            if args.save_array:
                all_data.append(x_sample.cpu().data.numpy()) # 256*4 x 8 x 3 x 16 x 128 x 128

    if args.save_array:
        print(f'saving numpy file to {save_path}')
        all_data_np = np.array(all_data)
        _, _, C, T, H, W = all_data_np.shape
        all_data_np = np.transpose(all_data_np.reshape(-1, C, T, H, W), (0, 2, 3, 4, 1))
        n_total = all_data_np.shape[0]
        np.save(os.path.join(save_path, 'samples_as_array.npy'), (all_data_np*255).astype(np.uint8)[np.random.permutation(n_total)[:args.n_sample]])