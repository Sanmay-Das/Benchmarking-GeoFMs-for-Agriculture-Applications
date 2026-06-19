"""
main_finetune_fpn.py
--------------------
SatMAE + SpectralGPT FPN decoder — crop segmentation fine-tuning.

Separate from main_finetune.py (PSANet decoder). Does NOT modify any
existing files. Uses the same dataset (dataset_seg.py) and evaluation
metrics (engine_finetune.py:compute_miou / evaluate_seg).

Key differences vs PSANet run:
  - Decoder  : SatMAEFPN (SpectralGPT-style FPN) instead of PSANet
  - Loss      : CrossEntropyLoss(ignore_index=0)  — properly masks NoData
  - Optimizer : AdamW  — better for from-scratch decoder heads
  - Output    : output_seg_Iowa_fpn/
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit_group_channels
from models_satmae_fpn import SatMAEFPN
from dataset_seg import build_seg_dataset
from engine_finetune import compute_miou


# ── Args ──────────────────────────────────────────────────────────────────────

def get_args_parser():
    parser = argparse.ArgumentParser('SatMAE FPN Segmentation', add_help=False)

    # Training
    parser.add_argument('--batch_size',   default=8,   type=int)
    parser.add_argument('--epochs',       default=100, type=int)
    parser.add_argument('--accum_iter',   default=16,  type=int)

    # Model
    parser.add_argument('--model',        default='vit_large_patch16', type=str)
    parser.add_argument('--input_size',   default=96,  type=int)
    parser.add_argument('--patch_size',   default=8,   type=int)
    parser.add_argument('--in_chans',     default=18,  type=int)
    parser.add_argument('--nb_classes',   default=14,  type=int)
    parser.add_argument('--drop_path',    default=0.2, type=float)
    parser.add_argument('--ignore_index', default=0,   type=int)

    # Band grouping
    parser.add_argument('--grouped_bands', type=int, nargs='+', action='append',
                        default=[])

    # Optimizer — SGD following SatMAE paper A.10 (same encoder, same protocol)
    parser.add_argument('--lr',            default=None,  type=float,
                        help='Head LR. Encoder gets 0.1x. Paper: 1e-2 head, 1e-3 encoder.')
    parser.add_argument('--blr',           default=1e-2,  type=float)
    parser.add_argument('--weight_decay',  default=1e-4,  type=float)
    parser.add_argument('--momentum',      default=0.9,   type=float)
    parser.add_argument('--min_lr',        default=1e-6,  type=float)
    parser.add_argument('--warmup_epochs', default=0,     type=int)
    parser.add_argument('--poly_power',    default=0.9,   type=float)

    # Pretrained weights
    parser.add_argument('--finetune', default='', type=str)

    # Dataset
    parser.add_argument('--data_path',
        default='/bigdata/eldawylab/sdas050/MS_Research/SatMAE_chips_multitemporal',
        type=str)
    parser.add_argument('--train_path',
        default='/bigdata/eldawylab/sdas050/MS_Research/SatMAE_chips_multitemporal/Iowa/train.txt',
        type=str)
    parser.add_argument('--test_path',
        default='/bigdata/eldawylab/sdas050/MS_Research/SatMAE_chips_multitemporal/Iowa/val.txt',
        type=str)

    # Output
    parser.add_argument('--output_dir', default='./output_seg_Iowa_fpn', type=str)
    parser.add_argument('--log_dir',    default='./output_seg_Iowa_fpn', type=str)
    parser.add_argument('--save_every', default=5,  type=int)

    # Runtime
    parser.add_argument('--device',       default='cuda')
    parser.add_argument('--seed',         default=42,  type=int)
    parser.add_argument('--num_workers',  default=8,   type=int)
    parser.add_argument('--pin_mem',      default=True, action='store_true')
    parser.add_argument('--resume',       default='',  type=str)
    parser.add_argument('--start_epoch',  default=0,   type=int)
    parser.add_argument('--eval',         action='store_true')

    # Distributed
    parser.add_argument('--world_size',  default=1,   type=int)
    parser.add_argument('--local_rank',  default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url',    default='env://')

    return parser


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, criterion, data_loader, optimizer, device,
                    epoch, loss_scaler, log_writer=None, args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    for step, (samples, targets) in enumerate(
            metric_logger.log_every(data_loader, 20, header)):

        samples = samples.to(device, non_blocking=True)   # (B, 18, 96, 96)
        targets = targets.to(device, non_blocking=True)   # (B, 96, 96)

        with torch.cuda.amp.autocast():
            logits = model(samples)                       # (B, nb_classes, 96, 96)
            loss   = criterion(logits, targets)

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            print(f'Loss is {loss_val}, stopping.')
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(step + 1) % accum_iter == 0)
        if (step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_val)
        metric_logger.update(lr=optimizer.param_groups[-1]['lr'])

        if log_writer is not None and (step + 1) % accum_iter == 0:
            epoch_1000x = int((step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('seg_fpn/train_loss', loss_val, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(data_loader, model, device, num_classes=13, ignore_index=0):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    metric_logger = misc.MetricLogger(delimiter='  ')
    model.eval()

    all_preds, all_targets = [], []
    total_loss, n_batches  = 0.0, 0

    for images, targets in metric_logger.log_every(data_loader, 10, 'Val:'):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            logits = model(images)
            loss   = criterion(logits, targets)

        preds = logits.argmax(dim=1)
        all_preds.append(preds)
        all_targets.append(targets)
        total_loss += loss.item()
        n_batches  += 1

        valid   = targets != ignore_index
        correct = (preds[valid] == targets[valid]).sum().item()
        total   = valid.sum().item()
        if total > 0:
            metric_logger.meters['acc1'].update(correct / total * 100, n=images.shape[0])
        metric_logger.update(loss=loss.item())

    all_preds   = torch.cat(all_preds,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    miou, per_class = compute_miou(all_preds, all_targets, num_classes, ignore_index)

    print(f'\n* Segmentation Results:')
    print(f'  mIoU: {miou*100:.2f}%')
    print(f'  OA:   {metric_logger.acc1.global_avg:.2f}%')
    print(f'  Loss: {total_loss/max(n_batches,1):.4f}')
    class_names = [
        'Natural Vegetation', 'Forest', 'Corn', 'Soybeans', 'Wetlands',
        'Developed/Barren', 'Open Water', 'Winter Wheat', 'Alfalfa',
        'Fallow/Idle', 'Cotton', 'Sorghum', 'Other',
    ]
    print('\n  Per-class IoU:')
    for i, (iou_val, name) in enumerate(zip(per_class, class_names), start=1):
        val_str = f'{iou_val*100:.2f}%' if not np.isnan(iou_val) else '  n/a'
        print(f'    {i:2d} {name:<22}: {val_str}')

    metric_logger.synchronize_between_processes()
    return {
        'miou': miou,
        'acc1': metric_logger.acc1.global_avg,
        'loss': total_loss / max(n_batches, 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    misc.init_distributed_mode(args)
    print('job dir:', os.path.dirname(os.path.realpath(__file__)))
    print(str(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    torch.manual_seed(args.seed + misc.get_rank())
    np.random.seed(args.seed + misc.get_rank())
    cudnn.benchmark = True

    # ── Dataset ───────────────────────────────────────────────────────────────
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
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False)

    # ── Band groups ───────────────────────────────────────────────────────────
    if len(args.grouped_bands) == 0:
        args.grouped_bands = [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
            [12, 13, 14, 15, 16, 17],
        ]
    print(f'Band groups: {args.grouped_bands}')

    # ── Build encoder ─────────────────────────────────────────────────────────
    encoder = models_vit_group_channels.__dict__[args.model](
        patch_size=args.patch_size,
        img_size=args.input_size,
        in_chans=args.in_chans,
        channel_groups=args.grouped_bands,
        num_classes=0,
        drop_path_rate=args.drop_path,
        global_pool=False,
    )

    # ── Load pretrained weights ───────────────────────────────────────────────
    if args.finetune:
        ckpt = torch.load(args.finetune, map_location='cpu')
        print(f'Loading pretrained weights: {args.finetune}')
        ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
        state_dict = encoder.state_dict()

        # Remove classification head keys
        for k in [k for k in ckpt_model if k.startswith(('head', 'fc_norm'))]:
            del ckpt_model[k]

        # Adapt patch_embed weights if channel counts differ
        num_groups = len(args.grouped_bands)
        for i in range(num_groups):
            w_key = f'patch_embed.{i}.proj.weight'
            if w_key in ckpt_model and w_key in state_dict:
                ckpt_w  = ckpt_model[w_key]
                model_w = state_dict[w_key]
                if ckpt_w.shape == model_w.shape:
                    print(f'  patch_embed.{i}: exact match')
                elif ckpt_w.shape[0] == model_w.shape[0] and ckpt_w.shape[2:] == model_w.shape[2:]:
                    C_ft  = model_w.shape[1]
                    new_w = ckpt_w.mean(dim=1, keepdim=True).expand(-1, C_ft, -1, -1).clone()
                    ckpt_model[w_key] = new_w
                    print(f'  patch_embed.{i}: adapted {ckpt_w.shape[1]}→{C_ft} ch')
                else:
                    del ckpt_model[w_key]
                    print(f'  patch_embed.{i}: incompatible shape, skipped')

        # Remove remaining shape mismatches
        for k in list(ckpt_model):
            if k in state_dict and ckpt_model[k].shape != state_dict[k].shape:
                print(f'  Shape mismatch, removing: {k}')
                del ckpt_model[k]

        interpolate_pos_embed(encoder, ckpt_model)
        msg = encoder.load_state_dict(ckpt_model, strict=False)
        print(f'Pretrained weights loaded. Missing: {msg.missing_keys[:5]}')

    # ── Build full model ───────────────────────────────────────────────────────
    model = SatMAEFPN(encoder=encoder, nb_classes=args.nb_classes)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_params/1e6:.2f}M')

    # ── Optimizer — SGD following SatMAE paper A.10 ──────────────────────────────
    # Same encoder, same protocol: enc LR=1e-3, head LR=1e-2, poly decay p=0.9
    eff_batch = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch / 256

    enc_lr  = args.lr * 0.1
    head_lr = args.lr
    print(f'Effective batch: {eff_batch}  |  Enc LR: {enc_lr:.2e}  |  Head LR: {head_lr:.2e}')

    enc_params  = [p for n, p in model.named_parameters()
                   if n.startswith('encoder') and p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith('encoder') and p.requires_grad]
    param_groups = [
        {'params': enc_params,  'lr': enc_lr,  'weight_decay': args.weight_decay},
        {'params': head_params, 'lr': head_lr, 'weight_decay': args.weight_decay},
    ]
    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum)

    # Polynomial LR decay — paper A.10: power=0.9
    poly_fn   = lambda epoch: (1.0 - epoch / args.epochs) ** args.poly_power
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[poly_fn, poly_fn])

    loss_scaler = NativeScaler()

    # ── Loss — CrossEntropy with ignore_index=0 (NoData masked) ──────────────
    criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    print(f'Loss: CrossEntropyLoss(ignore_index={args.ignore_index})')

    # ── Resume ────────────────────────────────────────────────────────────────
    misc.load_model(args=args, model_without_ddp=model,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    # ── DDP ───────────────────────────────────────────────────────────────────
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # ── Eval only ─────────────────────────────────────────────────────────────
    if args.eval:
        stats = evaluate(data_loader_val, model, device,
                         num_classes=13, ignore_index=args.ignore_index)
        print(f'mIoU: {stats["miou"]*100:.2f}%')
        return

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f'\nStart training for {args.epochs} epochs')
    start_time = time.time()
    best_miou  = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer, args=args)

        scheduler.step()
        print(f"  Enc LR: {scheduler.get_last_lr()[0]:.2e}  |  Head LR: {scheduler.get_last_lr()[1]:.2e}")

        # Evaluate
        val_stats = evaluate(data_loader_val, model, device,
                             num_classes=13, ignore_index=args.ignore_index)
        miou = val_stats['miou']
        print(f'mIoU on val: {miou*100:.2f}%')

        if miou > best_miou:
            best_miou = miou
            if args.output_dir and misc.is_main_process():
                misc.save_model(args=args, model=model,
                                model_without_ddp=model_without_ddp,
                                optimizer=optimizer, loss_scaler=loss_scaler,
                                epoch='best')
        print(f'Best mIoU so far: {best_miou*100:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('seg_fpn/val_miou', miou, epoch)
            log_writer.add_scalar('seg_fpn/val_loss', val_stats['loss'], epoch)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}':   v for k, v in val_stats.items()},
            'epoch': epoch,
        }
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time
    print(f'Training time: {datetime.timedelta(seconds=int(total_time))}')
    print(f'Best mIoU: {best_miou*100:.2f}%')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
