# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader

from tats import VideoDataset, load_transformer
from tats.utils import save_video_grid
from tats.modules.gpt import sample_with_past
from tats.utils import shift_dim

parser = argparse.ArgumentParser()

# pretrained checkpoints
parser.add_argument('--gpt_ckpt', type=str, default='', help='path to pretrained gpt checkpoint')
parser.add_argument('--vqgan_ckpt', type=str, default='', help='path to pretrained vqgan checkpoint')

# saving related arguments
parser.add_argument('--save_dir', type=str, default='../samples', help='top level directory where generated samples will be saved')
parser.add_argument('--save_name', type=str, default='demo', help='sub-directory name where generated samples will be saved')

# sampling related arguments
parser.add_argument('--top_k', type=int, default=2048, help='top-k filtering (only keep top-k tokens)')
parser.add_argument('--top_p', type=float, default=0.92, help='nucleus (top-p) filtering parameter')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers')

# auxiliaries
parser.add_argument('--save_array', action='store_true', help='whether to save the generated samples as np array for later analysis')
parser.add_argument('--compute_fvd', action='store_true', help='whether to compute fvd (frechet video distance)')

# arguments related to the conditioning videos
parser.add_argument('--ctx_video_dir', type=str, default='train', help='path to conditioning videos')
parser.add_argument('--ctx_len', type=int, default=4, help='number of temporal latents used for conditioning')
parser.add_argument('--sequence_length', type=int, default=16, help='number of frames in video clips used for conditioning')
parser.add_argument('--resolution', type=int, default=256, help='resolution of video clips used for conditioning')
parser.add_argument('--frame_rate', type=int, default=8, help='frame rate of videos used for conditioning')

args = parser.parse_args()

@torch.no_grad()
def sample(model, ctx, ctx_frames=8, steps=256, temperature=None, top_k=None, callback=None, top_p=None, latent_shape=(4, 16, 16), n_cond=0):

    index_sample = sample_with_past(ctx, model.transformer, steps=steps, sample_logits=True, top_k=top_k, callback=callback, temperature=temperature, 
                                    top_p=top_p, cutoff_ctx=False)
    index = index_sample.reshape([ctx.shape[0], *latent_shape])
    index = torch.clamp(index-n_cond, min=0, max=model.first_stage_model.n_codes-1)
    x_sample = model.first_stage_model.decode(index)
    x_sample = torch.clamp(x_sample, -0.5, 0.5) + 0.5
    x_sample[:, 0, ctx_frames:, :16, :16] = 1
    x_sample[:, 1, ctx_frames:, :16, :16] = 0
    x_sample[:, 2, ctx_frames:, :16, :16] = 0

    return x_sample

if __name__ == "__main__":

    # load pretrained GPT-video model
    gpt = load_transformer(args.gpt_ckpt, vqgan_ckpt=args.vqgan_ckpt).cuda().eval()

    save_path = os.path.join(args.save_dir, args.save_name)
    print(f'generating and saving video to {save_path}')
    os.makedirs(save_path, exist_ok=True)

    # load video data to be conditioned on
    dataset = VideoDataset(args.ctx_video_dir, args.sequence_length, train=True, resolution=args.resolution, frame_rate=args.frame_rate)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=None, shuffle=True)

    print(f'dataset size: {len(dataset)}, dataloader size: {len(dataloader)}')
    print(f'latent shape: {gpt.first_stage_model.latent_shape}')
    
    code_length = np.prod(gpt.first_stage_model.latent_shape)
    all_data = []

    with torch.no_grad():
        for i, vid_dict in enumerate(dataloader):

            vid_clip = vid_dict['video'].cuda()  # video to be used as context
            ctx = gpt.first_stage_model.encode(vid_clip)
            ctx = ctx[:, :args.ctx_len, :, :]  # take the first ctx_len temporal latents
            ctx = ctx.reshape(ctx.shape[0], -1)

            x_sample = sample(gpt, ctx=ctx, ctx_frames=2*args.ctx_len, steps=code_length-ctx.shape[1], temperature=1., top_k=args.top_k, top_p=args.top_p, 
                          latent_shape=gpt.first_stage_model.latent_shape)
            
            for j in range(ctx.shape[0]):
                save_video_grid(torch.stack((x_sample[j], vid_clip[j]+0.5)), os.path.join(save_path, f'sample_{i}_{j}.mp4'), 1)

            if args.save_array:
                all_data.append(x_sample.cpu().data.numpy())

    if args.save_array:
        print(f'saving numpy file to {save_path}')
        all_data_np = np.array(all_data)
        all_data_np = np.transpose(all_data_np.reshape(-1, 3, 16, 256, 256), (0, 2, 3, 4, 1))  # TODO: fix dimensions here
        n_total = all_data_np.shape[0]
        all_data_np = (all_data_np*255).astype(np.uint8)
        np.save(os.path.join(save_path, 'samples_as_array.npy'), all_data_np)  # save all?

    if args.compute_fvd:
        from tats.fvd.fvd import get_fvd_logits, frechet_distance, load_fvd_model, polynomial_mmd
        from tats import VideoData
        device = torch.device('cuda')
        i3d = load_fvd_model(device)
        data = VideoData(args)
        loader = data.train_dataloader()
        real_embeddings = []
        print('computing fvd embeddings for real videos')
        for batch in loader:
            real_embeddings.append(get_fvd_logits(shift_dim((batch['video']+0.5)*255, 1, -1).byte().data.numpy(), i3d=i3d, device=device))
            if len(real_embeddings)*args.batch_size >=2048: break
        print('concat fvd embeddings for real videos')
        real_embeddings = torch.cat(real_embeddings, 0)[:2048]
        print('computing fvd embeddings for fake videos')
        fake_embeddings = []
        n_batch = all_data_np.shape[0]//args.batch_size
        for i in range(n_batch):
            fake_embeddings.append(get_fvd_logits(all_data_np[i*args.batch_size:(i+1)*args.batch_size], i3d=i3d, device=device))
        print('caoncat fvd embeddings for fake videos')
        fake_embeddings = torch.cat(fake_embeddings, 0)[:2048]
        print('FVD = %.2f'%(frechet_distance(fake_embeddings, real_embeddings)))
        print('KVD = %.2f'%(polynomial_mmd(fake_embeddings.cpu(), real_embeddings.cpu())))