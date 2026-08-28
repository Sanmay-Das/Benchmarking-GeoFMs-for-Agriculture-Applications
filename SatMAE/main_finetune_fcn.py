# --------------------------------------------------------
# SatMAE Crop Segmentation Fine-tuning
# GroupChannels ViT-Large + Prithvi FCNHead
# --------------------------------------------------------

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(
        _os.path.join(_d, 'configs', 'paths.py')):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _os.path.join(_d, 'configs'))
from paths import DATA_ROOT, MSR_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS  # noqa: E402

import argparse
import datetime
import json
import numpy as np
import os
import time
import wandb
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit_group_channels
from models_satmae_fcn import SatMAEFCN

from dataset_seg import build_seg_dataset
from engine_finetune import (
    train_one_epoch_seg,
    evaluate_seg,
)


class CrossEntropyAuxLoss(nn.Module):
    """CrossEntropy for main + auxiliary FCNHead outputs with NoData masking."""
    def __init__(self, ignore_index=0, aux_weight=0.4):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.aux_weight = aux_weight

    def forward(self, preds, target):
        if isinstance(preds, (list, tuple)):
            return self.ce(preds[0], target) + self.aux_weight * self.ce(preds[1], target)
        return self.ce(preds, target)


# ============================================================================
# ARGS
# ============================================================================

def get_args_parser():
    parser = argparse.ArgumentParser('SatMAE + FCNHead Crop Segmentation', add_help=False)

    # Training
    parser.add_argument('--batch_size',  default=8,   type=int)
    parser.add_argument('--epochs',      default=100,  type=int)
    parser.add_argument('--accum_iter', default=16,   type=int)

    # Model
    parser.add_argument('--model',      default='vit_large_patch16', type=str)
    parser.add_argument('--input_size', default=96,  type=int)
    parser.add_argument('--patch_size', default=8,   type=int)
    parser.add_argument('--in_chans',   default=18,  type=int)
    parser.add_argument('--nb_classes', default=14,  type=int)
    parser.add_argument('--drop_path',  default=0.2, type=float)

    # Band grouping
    parser.add_argument('--grouped_bands', type=int, nargs='+', action='append',
                        default=[])

    # Loss
    parser.add_argument('--ignore_index', default=0, type=int)

    # Optimizer -- SGD following SatMAE paper A.10 (same encoder, same protocol)
    parser.add_argument('--lr',           default=None,  type=float,
                        help='Head LR. Encoder gets 0.1x. Paper: 1e-2 head, 1e-3 encoder.')
    parser.add_argument('--blr',          default=1e-2,  type=float)
    parser.add_argument('--weight_decay', default=1e-4,  type=float)
    parser.add_argument('--momentum',     default=0.9,   type=float)
    parser.add_argument('--min_lr',       default=1e-6,  type=float)
    parser.add_argument('--warmup_epochs',default=0,     type=int)
    parser.add_argument('--poly_power',   default=0.9,   type=float)

    # Pretrained weights
    parser.add_argument('--finetune', default='', help='Path to pretrained SatMAE checkpoint')

    # Dataset paths
    parser.add_argument('--data_path',
                        default=f'{DATA_ROOT}/SatMAE_chips_multitemporal',
                        type=str)
    parser.add_argument('--train_path',
                        default=f'{DATA_ROOT}/SatMAE_chips_multitemporal/Iowa/train.txt',
                        type=str)
    parser.add_argument('--test_path',
                        default=f'{DATA_ROOT}/SatMAE_chips_multitemporal/Iowa/val.txt',
                        type=str)

    # Output
    parser.add_argument('--output_dir', default='./output_seg_Iowa_fcn', type=str)
    parser.add_argument('--log_dir',    default='./output_seg_Iowa_fcn', type=str)
    parser.add_argument('--save_every', default=5, type=int)

    # Runtime
    parser.add_argument('--device',      default='cuda')
    parser.add_argument('--seed',        default=42, type=int)
    parser.add_argument('--num_workers', default=8,  type=int)
    parser.add_argument('--pin_mem',     default=True, action='store_true')
    parser.add_argument('--wandb',       default=None, type=str)
    parser.add_argument('--resume',      default='', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval',        action='store_true')

    # Distributed
    parser.add_argument('--world_size',  default=1, type=int)
    parser.add_argument('--local_rank',  default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url',    default='env://')

    return parser


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    dataset_train = build_seg_dataset(is_train=True,  args=args)
    dataset_val   = build_seg_dataset(is_train=False, args=args)

    num_tasks   = misc.get_world_size()
    global_rank = misc.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # -------------------------------------------------------------------------
    # Band groups
    # -------------------------------------------------------------------------
    if len(args.grouped_bands) == 0:
        args.grouped_bands = [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
            [12, 13, 14, 15, 16, 17],
        ]
    print(f"Band groups: {args.grouped_bands}")

    # -------------------------------------------------------------------------
    # Build encoder (GroupChannels ViT-Large)
    # -------------------------------------------------------------------------
    encoder = models_vit_group_channels.__dict__[args.model](
        patch_size=args.patch_size,
        img_size=args.input_size,
        in_chans=args.in_chans,
        channel_groups=args.grouped_bands,
        num_classes=0,
        drop_path_rate=args.drop_path,
        global_pool=False,
    )

    # -------------------------------------------------------------------------
    # Load pretrained weights
    # -------------------------------------------------------------------------
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print(f"Loading pretrained weights from: {args.finetune}")
        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state_dict = encoder.state_dict()

        for k in list(checkpoint_model.keys()):
            if k.startswith('head') or k.startswith('fc_norm'):
                del checkpoint_model[k]

        num_groups = len(args.grouped_bands)
        for i in range(num_groups):
            w_key = f'patch_embed.{i}.proj.weight'
            b_key = f'patch_embed.{i}.proj.bias'
            if w_key in checkpoint_model and w_key in state_dict:
                ckpt_w  = checkpoint_model[w_key]
                model_w = state_dict[w_key]
                if ckpt_w.shape == model_w.shape:
                    print(f"  patch_embed.{i}: exact match")
                elif (ckpt_w.shape[0] == model_w.shape[0] and
                      ckpt_w.shape[2:] == model_w.shape[2:]):
                    C_ft  = model_w.shape[1]
                    avg_w = ckpt_w.mean(dim=1, keepdim=True)
                    checkpoint_model[w_key] = avg_w.expand(-1, C_ft, -1, -1).clone()
                    print(f"  patch_embed.{i}: adapted {ckpt_w.shape[1]} -> {C_ft} ch")
                else:
                    del checkpoint_model[w_key]
                    print(f"  patch_embed.{i}: incompatible, skipping")
            if b_key in checkpoint_model and b_key in state_dict:
                if checkpoint_model[b_key].shape != state_dict[b_key].shape:
                    del checkpoint_model[b_key]

        for k in list(checkpoint_model.keys()):
            if k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"  Shape mismatch, removing: {k}")
                del checkpoint_model[k]

        interpolate_pos_embed(encoder, checkpoint_model)
        msg = encoder.load_state_dict(checkpoint_model, strict=False)
        print(f"Pretrained weights loaded. Missing: {msg.missing_keys}")

    # -------------------------------------------------------------------------
    # Build SatMAEFCN (encoder + FCNHead)
    # -------------------------------------------------------------------------
    model = SatMAEFCN(encoder=encoder, nb_classes=args.nb_classes)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of params (M): {n_parameters / 1e6:.2f}')

    # -------------------------------------------------------------------------
    # Optimizer -- SGD following SatMAE paper A.10
    # Same encoder, same protocol: enc LR=1e-3, head LR=1e-2, poly decay p=0.9
    # -------------------------------------------------------------------------
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    enc_lr  = args.lr * 0.1
    head_lr = args.lr
    print(f"Optimizer: SGD  |  Enc LR: {enc_lr:.2e}  |  Head LR: {head_lr:.2e}")

    enc_params  = [p for n, p in model_without_ddp.named_parameters()
                   if n.startswith('encoder') and p.requires_grad]
    head_params = [p for n, p in model_without_ddp.named_parameters()
                   if not n.startswith('encoder') and p.requires_grad]
    param_groups = [
        {'params': enc_params,  'lr': enc_lr,  'weight_decay': args.weight_decay},
        {'params': head_params, 'lr': head_lr, 'weight_decay': args.weight_decay},
    ]
    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum)

    # Polynomial LR decay -- paper A.10: power=0.9
    poly_fn   = lambda epoch: (1.0 - epoch / args.epochs) ** args.poly_power
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[poly_fn, poly_fn])

    loss_scaler = NativeScaler()

    # -------------------------------------------------------------------------
    # Loss -- CrossEntropy + ignore_index=0 (NoData masked)
    # -------------------------------------------------------------------------
    criterion = CrossEntropyAuxLoss(ignore_index=args.ignore_index, aux_weight=0.4)
    print(f"Criterion: CrossEntropyAuxLoss(ignore_index={args.ignore_index}, aux_weight=0.4)")

    # -------------------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------------------
    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    # -------------------------------------------------------------------------
    # DDP
    # -------------------------------------------------------------------------
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # -------------------------------------------------------------------------
    # WandB
    # -------------------------------------------------------------------------
    if global_rank == 0 and args.wandb is not None:
        wandb.init(project=args.wandb)
        wandb.config.update(args)
        wandb.watch(model)

    # -------------------------------------------------------------------------
    # Eval only
    # -------------------------------------------------------------------------
    if args.eval:
        test_stats = evaluate_seg(data_loader_val, model, device,
                                  num_classes=13,
                                  ignore_index=args.ignore_index)
        print(f"mIoU: {test_stats['miou']*100:.2f}%")
        exit(0)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_miou  = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch_seg(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer, args=args,
        )

        scheduler.step()
        print(f"  Enc LR: {scheduler.get_last_lr()[0]:.2e}  |  Head LR: {scheduler.get_last_lr()[1]:.2e}")

        test_stats = evaluate_seg(data_loader_val, model, device,
                                  num_classes=13,
                                  ignore_index=args.ignore_index)

        miou = test_stats['miou']
        print(f"mIoU on val: {miou*100:.2f}%")

        if miou > best_miou:
            best_miou = miou
            if args.output_dir and misc.is_main_process():
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch='best')
        print(f'Best mIoU so far: {best_miou*100:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('seg/val_miou', miou, epoch)
            log_writer.add_scalar('seg/val_loss', test_stats['loss'], epoch)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}':   v for k, v in test_stats.items()},
            'epoch': epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))
    print(f'Best mIoU: {best_miou*100:.2f}%')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
