import time
import os
import datetime
import torch
import argparse
from pathlib import Path

from src.models_vit_tensor_CD_2 import vit_base_patch8
from src.SpectralGPT_dataset import SpectralGPTSegDataset
from train_utils import train_one_epoch, evaluate, create_lr_scheduler, save_on_master, mkdir
import util.misc as misc
from util.pos_embed import interpolate_pos_embed


def create_model(nb_classes, weight_path=None, pretrain=False):
 """
 Create SpectralGPT model
 """
 model = vit_base_patch8(
 num_classes=nb_classes
 )
 
 if pretrain and weight_path and os.path.exists(weight_path):
 print(f"\n{'='*70}")
 print(f" LOADING PRETRAINED WEIGHTS")
 print(f"{'='*70}")
 
 checkpoint = torch.load(weight_path, map_location='cpu')
 print(f"From: {weight_path}")
 
 checkpoint_model = checkpoint['model']
 state_dict = model.state_dict()
 
 # Remove mismatched keys
 keys_to_remove = ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 
 'head.weight', 'head.bias', 'pos_embed_spatial', 'pos_embed_temporal',
 'cls_seg.0.weight', 'cls_seg.0.bias']
 
 for k in keys_to_remove:
 if k in checkpoint_model:
 if k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
 print(f" ️ Removing {k}: shape mismatch")
 print(f" Pretrained: {checkpoint_model[k].shape}")
 print(f" Current: {state_dict[k].shape}")
 del checkpoint_model[k]
 
 # Load weights
 msg = model.load_state_dict(checkpoint_model, strict=False)
 print(f"\n Loaded pretrained weights")
 print(f" Missing keys: {len(msg.missing_keys)}")
 print(f" Unexpected keys: {len(msg.unexpected_keys)}")
 else:
 print("\nTraining from scratch (no pretrained weights)")
 
 return model


def main(args):
 misc.init_distributed_mode(args)
 
 print("\n" + "="*70)
 print("SPECTRALGPT CUSTOM TRAINING")
 print("="*70)
 print(f"\nConfiguration:")
 print(f" Data path: {args.data_path}")
 print(f" Batch size: {args.batch_size}")
 print(f" Epochs: {args.epochs}")
 print(f" Learning rate: {args.lr}")
 print(f" Num classes: {args.num_classes + 1}")
 print(f" Device: {args.device}")

 device = torch.device(args.device)
 num_classes = args.num_classes + 1 # 13 classes (0-12) + background
 
 results_file = os.path.join(args.output_dir, "training_results.txt")

 # Create datasets
 print("\n" + "="*70)
 print("LOADING DATASETS")
 print("="*70)
 
 train_dataset = SpectralGPTSegDataset(
 args.data_path, 
 txt_name="train.txt", 
 training=True
 )
 
 val_dataset = SpectralGPTSegDataset(
 args.data_path, 
 txt_name="test.txt", 
 training=False
 )
 
 print(f"\n Train: {len(train_dataset)} samples")
 print(f"Val: {len(val_dataset)} samples")

 # Data loaders
 if args.distributed:
 train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
 test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
 else:
 train_sampler = torch.utils.data.RandomSampler(train_dataset)
 test_sampler = torch.utils.data.SequentialSampler(val_dataset)

 train_data_loader = torch.utils.data.DataLoader(
 train_dataset, 
 batch_size=args.batch_size,
 sampler=train_sampler, 
 num_workers=args.workers,
 collate_fn=train_dataset.collate_fn, 
 drop_last=True,
 pin_memory=True
 )

 val_data_loader = torch.utils.data.DataLoader(
 val_dataset, 
 batch_size=1,
 sampler=test_sampler, 
 num_workers=args.workers,
 collate_fn=val_dataset.collate_fn,
 pin_memory=True
 )

 # Create model
 print("\n" + "="*70)
 print("CREATING MODEL")
 print("="*70)
 
 model = create_model(
 nb_classes=num_classes, 
 weight_path=args.pretrain_path, 
 pretrain=args.use_pretrain
 )
 model.to(device)
 
 n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
 print(f"\n Model created: {n_parameters/1e6:.1f}M trainable parameters")

 if args.sync_bn:
 model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

 model_without_ddp = model
 if args.distributed:
 model = torch.nn.parallel.DistributedDataParallel(
 model, device_ids=[args.gpu], find_unused_parameters=True
 )
 model_without_ddp = model.module

 params_to_optimize = [p for p in model_without_ddp.parameters() if p.requires_grad]
 
 # Optimizer
 optimizer = torch.optim.AdamW(
 params_to_optimize,
 lr=args.lr,
 weight_decay=args.weight_decay
 )

 scaler = torch.cuda.amp.GradScaler() if args.amp else None

 lr_scheduler = create_lr_scheduler(
 optimizer, 
 len(train_data_loader), 
 args.epochs, 
 warmup=True,
 warmup_epochs=args.warmup_epochs
 )

 # Resume from checkpoint
 if args.resume:
 checkpoint = torch.load(args.resume, map_location='cpu')
 model_without_ddp.load_state_dict(checkpoint['model'])
 optimizer.load_state_dict(checkpoint['optimizer'])
 lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
 args.start_epoch = checkpoint['epoch'] + 1
 if args.amp and 'scaler' in checkpoint:
 scaler.load_state_dict(checkpoint["scaler"])
 print(f" Resumed from epoch {args.start_epoch}")

 if args.test_only:
 print("\n TEST ONLY MODE")
 confmat = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
 print(confmat)
 return

 # Training loop
 print("\n" + "="*70)
 print(" STARTING TRAINING")
 print("="*70)
 
 start_time = time.time()
 best_iou = 0.0
 
 for epoch in range(args.start_epoch, args.epochs):
 if args.distributed:
 train_sampler.set_epoch(epoch)
 
 print(f"\n{'='*70}")
 print(f"Epoch {epoch}/{args.epochs}")
 print(f"{'='*70}")
 
 mean_loss, lr = train_one_epoch(
 model, optimizer, train_data_loader, device, epoch,
 lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler
 )

 confmat = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
 val_info = str(confmat)
 print(val_info)
 
 # Get mIoU
 current_iou = confmat.mean_iou if hasattr(confmat, 'mean_iou') else 0.0

 # Save results
 if args.rank in [-1, 0]:
 with open(results_file, "a") as f:
 train_info = f"[epoch: {epoch}]\n" \
 f"train_loss: {mean_loss:.4f}\n" \
 f"lr: {lr:.8f}\n" \
 f"mIoU: {current_iou:.4f}\n"
 f.write(train_info + val_info + "\n\n")

 # Save checkpoint
 if args.output_dir:
 save_file = {
 'model': model_without_ddp.state_dict(),
 'optimizer': optimizer.state_dict(),
 'lr_scheduler': lr_scheduler.state_dict(),
 'args': args,
 'epoch': epoch,
 'mIoU': current_iou
 }
 if args.amp:
 save_file["scaler"] = scaler.state_dict()

 # Save best model
 if current_iou > best_iou:
 best_iou = current_iou
 save_on_master(save_file, os.path.join(args.output_dir, 'best_model.pth'))
 print(f" Saved best model (mIoU: {best_iou:.4f})")
 
 # Save latest
 save_on_master(save_file, os.path.join(args.output_dir, 'latest_model.pth'))
 
 # Save periodic checkpoint
 if (epoch + 1) % args.save_freq == 0:
 save_on_master(save_file, os.path.join(args.output_dir, f'model_epoch_{epoch}.pth'))

 total_time = time.time() - start_time
 total_time_str = str(datetime.timedelta(seconds=int(total_time)))
 print(f'\n Training complete!')
 print(f' Total time: {total_time_str}')
 print(f' Best mIoU: {best_iou:.4f}')


if __name__ == "__main__":
 parser = argparse.ArgumentParser(description='SpectralGPT Custom Training')

 # Data
 parser.add_argument('--data-path', 
 default=r"C:\MS_Research\SpectralGPT_chips\CentIA", 
 help='Path to chips directory')
 
 parser.add_argument('--device', default='cuda', help='device')
 
 # Model
 parser.add_argument('--num-classes', default=12, type=int, 
 help='Number of classes (0-12)')
 
 # Training
 parser.add_argument('-b', '--batch-size', default=32, type=int,
 help='Batch size (reduce if OOM)')
 
 parser.add_argument('--start_epoch', default=0, type=int)
 
 parser.add_argument("--pretrain-path", 
 default="../../weights/spectralgpt_base.pth",
 help="Path to pretrained weights")
 
 parser.add_argument("--use-pretrain", default=True, type=bool)
 
 parser.add_argument("--warmup-epochs", default=10, type=int)
 
 parser.add_argument('--epochs', default=150, type=int)
 
 parser.add_argument('--lr', default=5e-4, type=float)
 
 parser.add_argument('--weight-decay', default=1e-5, type=float)
 
 # System
 parser.add_argument('--sync_bn', type=bool, default=False)
 parser.add_argument('-j', '--workers', default=4, type=int)
 parser.add_argument('--print-freq', default=50, type=int)
 parser.add_argument('--save-freq', default=10, type=int,
 help='Save checkpoint every N epochs')
 parser.add_argument('--output-dir', default='./output_custom/')
 parser.add_argument('--resume', default='')
 parser.add_argument("--test-only", dest="test_only", action="store_true")
 
 # Distributed
 parser.add_argument('--world-size', default=1, type=int)
 parser.add_argument('--dist-url', default='env://')
 parser.add_argument('--dist_on_itp', action='store_true')
 parser.add_argument("--amp", default=False, type=bool)

 args = parser.parse_args()

 if args.output_dir:
 mkdir(args.output_dir)

 main(args)