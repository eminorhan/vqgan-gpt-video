# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tats import VQGAN, VideoData
from tats.modules.callbacks import ImageLogger, VideoLogger

def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VQGAN.add_model_specific_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    args = parser.parse_args()
    print(args)

    data = VideoData(args)

    # automatically adjust learning rate
    bs, base_lr, ngpu, accumulate = args.batch_size, args.lr, args.gpus, args.accumulate_grad_batches
    args.lr = accumulate * (ngpu/8.) * (bs/4.) * base_lr
    print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus/8) * {} (batchsize/4) * {:.2e} (base_lr)".format(args.lr, accumulate, ngpu/8, bs/4, base_lr))

    model = VQGAN(args)

    callbacks = []
    callbacks.append(ModelCheckpoint(every_n_train_steps=1000, save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    callbacks.append(ImageLogger(batch_frequency=1000, max_images=1, clamp=True))
    callbacks.append(VideoLogger(batch_frequency=1000, max_videos=1, clamp=True))

    kwargs = dict()
    if args.gpus > 1:
        kwargs = dict(strategy='ddp', gpus=args.gpus)

    # resuming from a checkpoint?
    if args.resume_from_checkpoint:
        print(f"will start from the recent ckpt {args.resume_from_checkpoint}")
    else:
        print(f"will start training from scratch")

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, max_steps=args.max_steps, **kwargs)
    trainer.fit(model, data)

if __name__ == '__main__':
    main()