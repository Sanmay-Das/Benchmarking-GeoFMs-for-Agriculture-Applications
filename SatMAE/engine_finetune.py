# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
import wandb
import torch.nn.functional as F
import numpy as np

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched


# ============================================================================
# CLASSIFICATION TRAINING (unchanged)
# ============================================================================

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log({'train_loss_step': loss_value_reduce,
                               'train_lr_step': max_lr, 'epoch_1000x': epoch_1000x})
                except ValueError:
                    pass

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ============================================================================
# SEGMENTATION TRAINING -- NEW
# ============================================================================

def train_one_epoch_seg(model: torch.nn.Module, criterion: torch.nn.Module,
                        data_loader: Iterable, optimizer: torch.optim.Optimizer,
                        device: torch.device, epoch: int, loss_scaler,
                        max_norm: float = 0, log_writer=None, args=None):
    """
    Training loop for segmentation (PSANet / FCNHead).
    DataLoader yields (image, mask).
    model.forward(x, is_train=True) returns (main_pred, aux_pred).
    criterion = CrossEntropyAuxLoss(ignore_index=0, aux_weight=0.4).
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device, non_blocking=True)   # (B, 18, 96, 96)
        targets = targets.to(device, non_blocking=True)   # (B, 96, 96)  long

        with torch.cuda.amp.autocast():
            main_pred, aux_pred = model(samples, is_train=True)
            loss = criterion([main_pred, aux_pred], targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        max_lr = 0.
        for group in optimizer.param_groups:
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('seg/loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('seg/lr', max_lr, epoch_1000x)
            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log({'seg_train_loss': loss_value_reduce,
                               'seg_lr': max_lr, 'epoch_1000x': epoch_1000x})
                except ValueError:
                    pass

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ============================================================================
# SEGMENTATION EVALUATION -- NEW
# ============================================================================

def compute_miou(preds: torch.Tensor, targets: torch.Tensor,
                 num_classes: int, ignore_index: int = 0):
    """
    Compute per-class IoU and mean IoU.

    Args:
        preds:        (B, H, W) int64 predicted class indices
        targets:      (B, H, W) int64 ground truth labels
        num_classes:  number of classes (13 for crop segmentation)
        ignore_index: label value to ignore (0 = NoData)

    Returns:
        miou:        scalar mean IoU (ignoring classes with no GT pixels)
        per_class:   (num_classes,) array of per-class IoU
    """
    iou_list = []
    preds   = preds.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()

    # Mask out ignore pixels
    valid = targets != ignore_index
    preds   = preds[valid]
    targets = targets[valid]

    for cls in range(1, num_classes + 1):  # classes 1..13
        pred_cls   = preds   == cls
        target_cls = targets == cls

        intersection = (pred_cls & target_cls).sum()
        union        = (pred_cls | target_cls).sum()

        if union == 0:
            iou_list.append(float('nan'))  # class absent in this batch
        else:
            iou_list.append(intersection / union)

    per_class = np.array(iou_list)
    miou = float(np.nanmean(per_class))
    return miou, per_class


@torch.no_grad()
def evaluate_seg(data_loader, model, device, num_classes=13, ignore_index=0):
    """
    Evaluation loop for PSANet segmentation.
    Reports: loss, mIoU, per-class IoU, overall accuracy.

    Returns dict with keys: loss, miou, acc1 (overall acc, for compat with save_model).
    """
    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test (seg):'

    model.eval()

    all_preds   = []
    all_targets = []
    total_loss  = 0.0
    n_batches   = 0

    for batch in metric_logger.log_every(data_loader, 10, header):
        images  = batch[0].to(device, non_blocking=True)   # (B, 18, 96, 96)
        targets = batch[1].to(device, non_blocking=True)   # (B, 96, 96)

        with torch.cuda.amp.autocast():
            # PSANet returns main logits only during eval
            logits = model(images, is_train=False)          # (B, 13, 96, 96)
            loss   = criterion(logits, targets)

        preds = logits.argmax(dim=1)  # (B, 96, 96)

        all_preds.append(preds)
        all_targets.append(targets)
        total_loss += loss.item()
        n_batches  += 1

        # Overall pixel accuracy (ignoring nodata)
        valid   = targets != ignore_index
        correct = (preds[valid] == targets[valid]).sum().item()
        total   = valid.sum().item()
        if total > 0:
            batch_acc = correct / total * 100.0
            metric_logger.meters['acc1'].update(batch_acc, n=images.shape[0])

        metric_logger.update(loss=loss.item())

    # Compute global mIoU across all batches
    all_preds   = torch.cat(all_preds,   dim=0)  # (N, 96, 96)
    all_targets = torch.cat(all_targets, dim=0)  # (N, 96, 96)
    miou, per_class_iou = compute_miou(all_preds, all_targets, num_classes, ignore_index)

    metric_logger.synchronize_between_processes()

    # Print results
    class_names = [
        'Natural Vegetation', 'Forest', 'Corn', 'Soybeans', 'Wetlands',
        'Developed/Barren', 'Open Water', 'Winter Wheat', 'Alfalfa',
        'Fallow/Idle', 'Cotton', 'Sorghum', 'Other'
    ]
    print(f'\n* Segmentation Results:')
    print(f'  mIoU: {miou*100:.2f}%')
    print(f'  OA:   {metric_logger.acc1.global_avg:.2f}%')
    print(f'  Loss: {metric_logger.loss.global_avg:.4f}')
    print(f'\n  Per-class IoU:')
    for i, (name, iou) in enumerate(zip(class_names, per_class_iou)):
        if not np.isnan(iou):
            print(f'    {i+1:2d} {name:22s}: {iou*100:.2f}%')
        else:
            print(f'    {i+1:2d} {name:22s}: N/A (absent)')

    return {
        'loss':  metric_logger.loss.global_avg,
        'miou':  miou,
        'acc1':  miou * 100.0,   # use mIoU as acc1 for checkpoint saving compatibility
        'acc5':  metric_logger.acc1.global_avg,  # overall pixel acc as acc5
        'per_class_iou': per_class_iou.tolist(),
    }


# ============================================================================
# CLASSIFICATION EVALUATION (unchanged)
# ============================================================================

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_temporal(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images     = batch[0]
        timestamps = batch[1]
        target     = batch[-1]

        images     = images.to(device, non_blocking=True)
        timestamps = timestamps.to(device, non_blocking=True)
        target     = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images, timestamps)
            loss   = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}