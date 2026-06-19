"""
train_cd_spectralgpt.py
-----------------------
Training script for SpectralGPT Change Detection benchmark.

Mirrors your SegMunich/train_custom_spectralgpt.py setup exactly:
    - Same import paths (train_utils/, util/, src/)
    - Distributed training (single GPU on HPCC)
    - AMP mixed precision
    - AdamW + warmup LR scheduler
    - NLLLoss with class weights (handles changed/unchanged imbalance)
    - Saves best_F1 and last_checkpoint
    - Reports Precision, Recall, F1 (standard CD metrics per SpectralGPT paper)

Geographic split:
    Train : CentIA
    Val   : NWIA
    Test  : EastIA (run with --test-only)

Folder structure expected:
    downstream_tasks/ChangeDetection/
        src/
            model_cd_spectralgpt.py
        train_utils/         <- copied from SegMunich/train_utils/
        util/                <- copied from SegMunich/util/
        dataset_cd.py
        train_cd_spectralgpt.py   <- THIS FILE

Run:
    cd /bigdata/eldawylab/sdas050/MS_Research/IEEE_TPAMI_SpectralGPT/downstream_tasks/ChangeDetection

    python -m torch.distributed.launch \
        --nproc_per_node=1 --master_port=25643 --use_env \
        train_cd_spectralgpt.py \
        --data-root /bigdata/eldawylab/sdas050/MS_Research/change_detection_chips/spectralgpt \
        --pretrain-path /bigdata/eldawylab/sdas050/MS_Research/weights/SpectralGPT+.pth \
        --output-dir ./cd_train_spectralgpt

Test only (EastIA):
    python -m torch.distributed.launch \
        --nproc_per_node=1 --master_port=25643 --use_env \
        train_cd_spectralgpt.py \
        --data-root ... \
        --pretrain-path ... \
        --output-dir ./cd_train_spectralgpt \
        --resume ./cd_train_spectralgpt/best_F1_model.pth \
        --test-only
"""

import time
import os
import datetime
import torch
import torch.nn as nn
import argparse
import numpy as np

# Same import pattern as SegMunich/train_custom_spectralgpt.py
from src.model_cd_spectralgpt import build_spectralgpt_cd
from train_utils import create_lr_scheduler, save_on_master, mkdir
from dataset_cd import CDDataset
import util.misc as misc


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(tp, fp, tn, fn):
    """Precision, Recall, F1 for the changed class (class 1)."""
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    oa   = (tp + tn) / (tp + fp + tn + fn + 1e-8)
    return float(prec), float(rec), float(f1), float(oa)


# ============================================================================
# Train one epoch
# ============================================================================

def train_one_epoch_cd(model, optimizer, loader, device, epoch,
                        lr_scheduler, scaler, print_freq, criterion):
    model.train()
    tot_loss  = 0.0
    n_batches = 0

    for i, batch in enumerate(loader):
        t1   = batch['t1'].to(device)    # (B, 6, H, W)
        t2   = batch['t2'].to(device)
        mask = batch['mask'].to(device)  # (B, H, W) values: 0, 1, 255

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(t1, t2)           # (B, 2, H, W)
                loss   = criterion(output, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(t1, t2)
            loss   = criterion(output, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        lr_scheduler.step()

        tot_loss  += loss.item()
        n_batches += 1

        if i % print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch [{epoch}]  [{i}/{len(loader)}]  "
                  f"loss: {loss.item():.4f}  lr: {lr:.8f}")

    mean_loss = tot_loss / max(n_batches, 1)
    return mean_loss, optimizer.param_groups[0]['lr']


# ============================================================================
# Evaluate
# ============================================================================

@torch.no_grad()
def evaluate_cd(model, loader, device, criterion):
    model.eval()
    tot_loss  = 0.0
    n_batches = 0
    tp = fp = tn = fn = 0

    for batch in loader:
        t1   = batch['t1'].to(device)
        t2   = batch['t2'].to(device)
        mask = batch['mask'].to(device)

        output = model(t1, t2)                   # (B, 2, H, W)
        loss   = criterion(output, mask)
        tot_loss  += loss.item()
        n_batches += 1

        pred  = output.argmax(dim=1)             # (B, H, W)
        valid = (mask != 255)                    # ignore nodata

        pr = pred[valid].cpu().numpy()
        gt = mask[valid].cpu().numpy()

        tp += int(np.logical_and(pr == 1, gt == 1).sum())
        fp += int(np.logical_and(pr == 1, gt == 0).sum())
        tn += int(np.logical_and(pr == 0, gt == 0).sum())
        fn += int(np.logical_and(pr == 0, gt == 1).sum())

    prec, rec, f1, oa = compute_metrics(tp, fp, tn, fn)
    avg_loss = tot_loss / max(n_batches, 1)

    print(f"  Loss     : {avg_loss:.4f}")
    print(f"  OA       : {oa*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall   : {rec*100:.2f}%")
    print(f"  F1       : {f1*100:.2f}%")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    return avg_loss, prec, rec, f1, oa


# ============================================================================
# Main
# ============================================================================

def main(args):
    misc.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # ── datasets ──────────────────────────────────────────────────────
    train_csv = [os.path.join(args.data_root, 'CentIA_chips.csv')]   # train
    val_csv   = [os.path.join(args.data_root, 'EastIA_chips.csv')]   # val
    test_csv  = [os.path.join(args.data_root, 'NWIA_chips.csv')]     # test

    train_dataset = CDDataset(train_csv, training=True)
    val_dataset   = CDDataset(val_csv,   training=False)
    test_dataset  = CDDataset(test_csv,  training=False)

    print("Creating data loaders")

    # ── samplers ──────────────────────────────────────────────────────
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler   = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler   = torch.utils.data.SequentialSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=CDDataset.collate_fn, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1,
        sampler=val_sampler, num_workers=args.workers,
        collate_fn=CDDataset.collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1,
        sampler=torch.utils.data.SequentialSampler(test_dataset),
        num_workers=args.workers,
        collate_fn=CDDataset.collate_fn,
    )

    # ── model ─────────────────────────────────────────────────────────
    print("Creating model")
    model = build_spectralgpt_cd(
        pretrain_path=args.pretrain_path if not args.no_pretrain else None
    )
    model.to(device)

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # ── loss ──────────────────────────────────────────────────────────
    # Class weights from training masks; ignore_index=255 for nodata pixels
    weights   = torch.FloatTensor(train_dataset.weights).to(device)
    criterion = nn.NLLLoss(weight=weights, ignore_index=255)
    print(f"Class weights: unchanged={weights[0]:.4f}, changed={weights[1]:.4f}")

    # ── optimizer + scheduler ─────────────────────────────────────────
    params_to_optimize = [p for p in model_without_ddp.parameters()
                          if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = create_lr_scheduler(
        optimizer, len(train_loader), args.epochs,
        warmup=(args.warmup_epochs > 0), warmup_epochs=args.warmup_epochs,
    )

    # ── resume ────────────────────────────────────────────────────────
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        print(f"Resumed from epoch {checkpoint['epoch']}")

    # ── test only ─────────────────────────────────────────────────────
    if args.test_only:
        print("\n=== TEST SET (NWIA) ===")
        evaluate_cd(model, test_loader, device, criterion)
        return

    # ── training loop ─────────────────────────────────────────────────
    best_f1      = 0.0
    results_file = "cd_results_spectralgpt.txt"

    print("Start training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        mean_loss, lr = train_one_epoch_cd(
            model, optimizer, train_loader, device, epoch,
            lr_scheduler, scaler, args.print_freq, criterion,
        )

        print(f"\n=== Epoch {epoch} — Val (EastIA) ===") 
        val_loss, prec, rec, f1, oa = evaluate_cd(
            model, val_loader, device, criterion)

        improved = f1 > best_f1

        # logging — same pattern as SegMunich train.py
        if args.rank in [-1, 0]:
            with open(results_file, 'a') as f:
                f.write(
                    f"\n[epoch: {epoch}]\n"
                    f"train_loss: {mean_loss:.4f}\n"
                    f"lr: {lr:.8f}\n"
                    f"val_loss:   {val_loss:.4f}\n"
                    f"OA:         {oa*100:.2f}%\n"
                    f"Precision:  {prec*100:.2f}%\n"
                    f"Recall:     {rec*100:.2f}%\n"
                    f"F1:         {f1*100:.2f}%\n"
                )

        # save best F1 checkpoint
        if args.output_dir and improved:
            best_f1   = f1
            save_file = {
                'model':        model_without_ddp.state_dict(),
                'optimizer':    optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args':         args,
                'epoch':        epoch,
                'best_f1':      best_f1,
            }
            if args.amp:
                save_file['scaler'] = scaler.state_dict()
            save_on_master(save_file,
                           os.path.join(args.output_dir, 'best_F1_model.pth'))
            print(f"NEW BEST! Saved: best_F1_model.pth "
                  f"(Epoch {epoch}, F1: {best_f1*100:.2f}%)")

        # periodic checkpoint every 50 epochs
        if args.output_dir and (epoch % 50 == 0 or epoch == args.epochs - 1):
            save_file = {
                'model':        model_without_ddp.state_dict(),
                'optimizer':    optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args':         args,
                'epoch':        epoch,
                'best_f1':      best_f1,
            }
            if args.amp:
                save_file['scaler'] = scaler.state_dict()
            save_on_master(save_file,
                           os.path.join(args.output_dir, 'last_checkpoint.pth'))
            print(f"Checkpoint saved: last_checkpoint.pth (Epoch {epoch})")

    total_time = time.time() - start_time
    print(f"Training time: {datetime.timedelta(seconds=int(total_time))}")

    # final test on NWIA using best checkpoint
    print("\n=== FINAL TEST SET (NWIA) ===")
    best_ckpt = torch.load(
        os.path.join(args.output_dir, 'best_F1_model.pth'), map_location='cpu')
    model_without_ddp.load_state_dict(best_ckpt['model'])
    evaluate_cd(model, test_loader, device, criterion)


# ============================================================================
# Args — mirrors SegMunich/train_custom_spectralgpt.py argument style
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SpectralGPT Change Detection Training')

    parser.add_argument(
        '--data-root',
        default='/bigdata/eldawylab/sdas050/MS_Research/change_detection_chips/spectralgpt',
        help='folder containing CentIA_chips.csv, NWIA_chips.csv, EastIA_chips.csv',
    )
    parser.add_argument(
        '--pretrain-path',
        default='/bigdata/eldawylab/sdas050/MS_Research/weights/SpectralGPT+.pth',
    )
    parser.add_argument(
        '--no-pretrain', action='store_true',
        help='train from scratch (ablation only)',
    )
    parser.add_argument('--output-dir', default='./cd_train_spectralgpt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=60, type=int,
                        help='60 epochs matches SpectralGPT paper Section 3.4')
    parser.add_argument('--warmup-epochs', default=5, type=int)
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='1e-3 matches SpectralGPT paper Section 3.4')
    parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                        metavar='W', dest='weight_decay')
    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('--print-freq', default=50, type=int)
    parser.add_argument('--sync_bn', type=bool, default=False)
    parser.add_argument('--amp', default=True, type=bool)
    parser.add_argument('--resume', default='')
    parser.add_argument('--test-only', dest='test_only',
                        action='store_true')
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--dist-url', default='env://')
    parser.add_argument('--dist_on_itp', action='store_true')

    args = parser.parse_args()

    if args.output_dir:
        mkdir(args.output_dir)

    main(args)