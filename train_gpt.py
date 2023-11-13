# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tats import Net2NetTransformer, VideoData

def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Net2NetTransformer.add_model_specific_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    args = parser.parse_args()
    print(args)

    data = VideoData(args)

    args.class_cond_dim = data.n_classes if not args.unconditional and args.cond_stage_key=='label' else None
    model = Net2NetTransformer(args, first_stage_key=args.first_stage_key, cond_stage_key=args.cond_stage_key)

    callbacks = []
    callbacks.append(ModelCheckpoint(every_n_train_steps=1000, save_top_k=-1, filename='{epoch}-{step}-{train/loss:.2f}'))
    callbacks.append(ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=3, filename='best_checkpoint'))

    kwargs = dict()
    if args.gpus > 1:
        # find_unused_parameters = False to support gradient checkpointing
        kwargs = dict(gpus=args.gpus, plugins=[pl.plugins.DDPPlugin(find_unused_parameters=False)])

    # configure learning rate
    bs, base_lr = args.batch_size, args.base_lr
    ngpu = args.gpus
    accumulate_grad_batches = args.accumulate_grad_batches or 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
    print("Setting lr to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))

    # resuming from a checkpoint?
    if args.resume_from_checkpoint:
        print(f"will start from the recent ckpt {args.resume_from_checkpoint}")
    else:
        print(f"will start training from scratch")

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, max_steps=args.max_steps, **kwargs)
    trainer.fit(model, data)

if __name__ == '__main__':
    main()