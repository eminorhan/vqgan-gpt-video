# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader

from tats import VideoDataset, load_transformer
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
parser.add_argument('--num_workers', type=int, default=16)

# arguments related to the conditioning videos
parser.add_argument('--ctx_video_dir', type=str, default='test')
parser.add_argument('--ctx_len', type=int, default=4)
parser.add_argument('--sequence_length', type=int, default=16)
parser.add_argument('--resolution', type=int, default=256)
parser.add_argument('--frame_rate', type=int, default=8)

parser.add_argument('--sample_length', type=int, default=256)
parser.add_argument('--sample_resolution', type=int, default=16)
parser.add_argument('--temporal_sample_pos', type=int, default=1)
parser.add_argument('--save_array', action='store_true')

args = parser.parse_args()


@torch.no_grad()
def sample_long_fast(model, ctx, temporal_infer, spatial_infer, temporal_sample_pos, batch_size, temperature=1.):
    
    steps = slice_n_code = spatial_infer**2

    with torch.no_grad():
        index_sample_all = torch.zeros([batch_size, temporal_infer*spatial_infer*spatial_infer]).long().cuda()
        index_sample_all[:,:temporal_sample_pos*steps] = sample_with_past(ctx, model.transformer, steps=temporal_sample_pos*steps, sample_logits=True, 
                                                                          top_k=args.top_k, temperature=temperature, top_p=args.top_p, cutoff_ctx=False)
        
        for t_id in range(temporal_infer-temporal_sample_pos):
            i_start = t_id*slice_n_code
            i_end = (temporal_sample_pos+t_id)*slice_n_code
            x_past = index_sample_all[:,i_start:i_end]
            index_sample_all[:,i_end:i_end+steps] = sample_with_past(torch.cat([ctx, x_past], dim=1), model.transformer, steps=steps, sample_logits=True, 
                                                                     top_k=args.top_k, temperature=temperature, top_p=args.top_p, cutoff_ctx=False)
        
        torch.cuda.empty_cache()
        index_sample_all = index_sample_all.reshape([batch_size, temporal_infer, spatial_infer, spatial_infer])
        index_sample_all = torch.clamp(index_sample_all-model.cond_stage_vocab_size, min=0, max=model.first_stage_model.n_codes-1)
    
        x_sample = []
        for i in range(batch_size):
            x_sample.append(model.first_stage_model.decode(index_sample_all[i:i+1, :, :, :].cuda()).cpu())

        x_sample = torch.cat(x_sample, 0)
        x_sample = torch.clamp(x_sample, -0.5, 0.5) + 0.5

    return x_sample


if __name__ == "__main__":

    # load pretrained GPT-video model
    gpt = load_transformer(args.gpt_ckpt, vqgan_ckpt=args.vqgan_ckpt).cuda().eval()

    temporal_train, spatial_train, _ = gpt.first_stage_model.latent_shape
    all_data = []

    save_path = os.path.join(args.save_dir, args.save_name)
    print(f'generating and saving video to {save_path}')
    os.makedirs(save_path, exist_ok=True)

    # load video data to be conditioned on
    dataset = VideoDataset(args.ctx_video_dir, args.sequence_length, train=True, resolution=args.resolution, frame_rate=args.frame_rate)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=None, shuffle=True)

    with torch.no_grad():
        for i, vid_dict in enumerate(dataloader):

            vid_clip = vid_dict['video'].cuda()  # video to be used as context
            ctx = gpt.first_stage_model.encode(vid_clip)
            ctx = ctx[:, :args.ctx_len, :, :]  # take the first ctx_len temporal latents
            ctx = ctx.reshape(ctx.shape[0], -1)

            x_sample = sample_long_fast(gpt, ctx, args.sample_length, args.sample_resolution, temporal_sample_pos=args.temporal_sample_pos, batch_size=args.batch_size)

            for j in range(ctx.shape[0]):
                save_video_grid(torch.stack((x_sample[j], vid_clip[j]+0.5)), os.path.join(save_path, f'sample_{i}_{j}.mp4'), 1)

            if args.save_array:
                all_data.append(x_sample.cpu().data.numpy())

    if args.save_array:
        print(f'saving numpy file to {save_path}')
        all_data_np = np.array(all_data)
        _, _, C, T, H, W = all_data_np.shape
        all_data_np = np.transpose(all_data_np.reshape(-1, C, T, H, W), (0, 2, 3, 4, 1))
        n_total = all_data_np.shape[0]
        np.save(os.path.join(save_path, 'samples_as_array.npy'), (all_data_np*255).astype(np.uint8)[np.random.permutation(n_total)[:args.n_sample]])