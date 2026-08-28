import time
import os
import datetime
import torch
import re # ADD THIS LINE AT THE TOP

from src.models_vit_tensor_CD_2 import vit_base_patch8
from src import UNet
# from src.models_vit_group_channels_seg import vit_base_patch16
from train_utils import train_one_epoch, evaluate, create_lr_scheduler, init_distributed_mode, save_on_master, mkdir
#from TUM_128 import SegDataset
from SpectralGPTsegdataset import SegDataset
import transforms as T
import argparse
import util.misc as misc
import timm
import util.lr_decay as lrd
from util.pos_embed import interpolate_pos_embed
# os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

#CentIA
def create_model(nb_classes, weight_path, pretrain=False):
 model = vit_base_patch8(num_classes=nb_classes)
 # model = vit_large_patch8(num_classes=nb_classes)

 if pretrain:
 checkpoint = torch.load(weight_path, map_location='cpu')
 print("Load pre-trained checkpoint from: %s" % weight_path)
 checkpoint_model = checkpoint['model']
 # checkpoint_model = checkpoint
 state_dict = model.state_dict()
 
 # UPDATED: Add pos_embed_temporal and fc layers to removal list
 keys_to_check = [
 'pos_embed', 
 'pos_embed_spatial', 
 'pos_embed_temporal', # ADDED: Will mismatch (4 vs 6)
 'patch_embed.proj.weight', 
 'patch_embed.proj.bias', 
 'head.weight', 
 'head.bias',
 'fc.0.weight', # ADDED: Will mismatch (4 vs 6)
 'fc.0.bias' # ADDED: Will mismatch (4 vs 6)
 ]
 
 for k in keys_to_check:
 if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
 print(f"Removing key {k} from pretrained checkpoint")
 print(f" Pretrained shape: {checkpoint_model[k].shape}")
 print(f" Current shape: {state_dict[k].shape}")
 del checkpoint_model[k]
 
 interpolate_pos_embed(model, checkpoint_model)

 # load pre-trained model
 msg = model.load_state_dict(checkpoint_model, strict=False)
 print(msg)
 return model

def create_model_Unet(num_classes, weights, pretrain=False):
 model = UNet(in_channels=12, num_classes=num_classes, base_c=64)
 # model = UPerNet(num_classes=13)
 if pretrain:
 model.load_weights(weights)
 return model

def main(args):
 misc.init_distributed_mode(args)
 print(args)

 device = torch.device(args.device)
 # segmentation nun_classes + background
 batch_size = args.batch_size
 # segmentation nun_classes + background
 num_classes = args.num_classes + 1
 pretrain_path = args.pretrain_path
 warmup_epochs = args.warmup_epochs
 # coco_info
 # results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
 results_file = "vit_IL.txt"
 # results_file = "unet.txt"

 data_root = args.data_path
 # check data root
 # train_dataset = SegDataset(args.data_path, txt_name="train.txt", training=True, data_name="BigEarthNet")
 # val_dataset = SegDataset(args.data_path, txt_name="test.txt", training=False, data_name="BigEarthNet")

 train_dataset = SegDataset(args.data_path, txt_name="train.txt", training=True, data_name="MinnesotaChips")
 val_dataset = SegDataset(args.data_path, txt_name="test.txt", training=False, data_name="MinnesotaChips")

 print("Creating data loaders")
 if args.distributed:
 train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
 test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
 else:
 train_sampler = torch.utils.data.RandomSampler(train_dataset)
 test_sampler = torch.utils.data.SequentialSampler(val_dataset)

 train_data_loader = torch.utils.data.DataLoader(
 train_dataset, batch_size=batch_size,
 # prefetch_factor=0,
 sampler=train_sampler, num_workers=args.workers,
 collate_fn=train_dataset.collate_fn, drop_last=True)

 val_data_loader = torch.utils.data.DataLoader(
 val_dataset, batch_size=1,
 sampler=test_sampler, num_workers=args.workers,

 collate_fn=train_dataset.collate_fn)

 print("Creating model")
 # create model num_classes equal background + foreground classes
 # model = create_model_Unet(num_classes=num_classes, weights=pretrain_path, pretrain=False)
 model = create_model(nb_classes=num_classes, weight_path=pretrain_path, pretrain=True)
 # model = create_model(nb_classes=13, weight_path='./src/checkpoint-150.pth', pretrain=True)
 model.to(device)

 if args.sync_bn:
 model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

 model_without_ddp = model
 if args.distributed:
 model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
 model_without_ddp = model.module

 params_to_optimize = [p for p in model_without_ddp.parameters() if p.requires_grad]
 # param_groups = model_without_ddp.parameters()
 optimizer = torch.optim.AdamW(
 params_to_optimize,
 lr=args.lr, weight_decay=1e-5
 )
 # optimizer = torch.optim.AdamW(
 # param_groups,
 # lr=args.lr,weight_decay=0.01)
 # optimizer = torch.optim.SGD(
 # param_groups,
 # lr=args.lr,momentum=0.9,weight_decay=args.weight_decay)


 scaler = torch.cuda.amp.GradScaler() if args.amp else None

 # step(epoch)
 lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs, warmup=True,
 warmup_epochs=warmup_epochs)

 # resume
 if args.resume:
 # If map_location is missing, torch.load will first load the module to CPU
 # and then copy each parameter to where it was saved,
 # which would result in all processes on the same machine using the same set of devices.
 checkpoint = torch.load(args.resume, map_location='cpu') # ()
 model_without_ddp.load_state_dict(checkpoint['model'])
 optimizer.load_state_dict(checkpoint['optimizer'])
 lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
 args.start_epoch = checkpoint['epoch'] + 1
 if args.amp:
 scaler.load_state_dict(checkpoint["scaler"])

 if args.test_only:
 confmat = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
 val_info = str(confmat)
 print(val_info)
 return

 # NEW: Initialize best accuracy tracker
 best_miou = 0.
 
 print("Start training")
 start_time = time.time()
 for epoch in range(args.start_epoch, args.epochs):
 if args.distributed:
 train_sampler.set_epoch(epoch)
 mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch,
 lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

 confmat = evaluate(model, val_data_loader, device=device, num_classes=num_classes)
 val_info = str(confmat)
 print(val_info)
 
 # NEW: Extract OA from string output (format: "OA: 74.5")
 miou_match = re.search(r'mean IoU:\s*([\d.]+)', val_info)
 if miou_match:
 current_miou = float(miou_match.group(1)) / 100.0
 else:
 current_miou = 0.0

 improved = current_miou > best_miou

 # 
 if args.rank in [-1, 0]:
 # write into txt
 with open(results_file, "a") as f:
 # epochtrain_losslr
 train_info = f"[epoch: {epoch}]\n" \
 f"train_loss: {mean_loss:.4f}\n" \
 f"lr: {lr:.8f}\n" \
 f"mIoU: {current_miou*100:.2f}%\n" # NEW: Added OA
 f.write(train_info + val_info + "\n\n")

 # UPDATED: Save only if best
 if args.output_dir and improved:
 best_miou = current_miou # NEW: Update best
 
 # 
 save_file = {'model': model_without_ddp.state_dict(),
 'optimizer': optimizer.state_dict(),
 'lr_scheduler': lr_scheduler.state_dict(),
 'args': args,
 'epoch': epoch,
 'best_miou': current_miou} # NEW: Store best_oa
 if args.amp:
 save_file["scaler"] = scaler.state_dict()

 save_on_master(save_file,
 os.path.join(args.output_dir, 'best_mIoU_NorthCentMN_model.pth')) # CHANGED: Always save to best_NorthCentMN_model.pth
 print(f" NEW BEST! Saved: best_mIoU_NorthCentMN_model.pth (Epoch {epoch}, OA: {current_miou*100:.2f}%)") # NEW: Print message

 if args.output_dir and (epoch % 50 == 0 or epoch == args.epochs - 1):
 save_file = {
 'model': model_without_ddp.state_dict(),
 'optimizer': optimizer.state_dict(),
 'lr_scheduler': lr_scheduler.state_dict(),
 'args': args,
 'epoch': epoch,
 'best_miou': best_miou # Keep track of best_oa
 }
 if args.amp:
 save_file["scaler"] = scaler.state_dict()
 
 # Save as last_checkpoint.pth (overwrites each time)
 save_on_master(save_file,
 os.path.join(args.output_dir, 'last_checkpoint_NorthCentMN.pth'))
 print(f" Checkpoint saved: last_checkpoint_NorthCentMN.pth (Epoch {epoch})")

 total_time = time.time() - start_time
 total_time_str = str(datetime.timedelta(seconds=int(total_time)))
 print('Training time {}'.format(total_time_str))


if __name__ == "__main__":


 parser = argparse.ArgumentParser(
 description=__doc__)

 # (DRIVE)
 # parser.add_argument('--data-path', default="/home/ps/Documents/data/TUM_128", help='dataset')
 # CentIA single date
 # parser.add_argument('--data-path', default="/bigdata/eldawylab/sdas050/MS_Research/SpectralGPT_chips/CentIA", help='dataset')
 # IA multi-temporal
 parser.add_argument('--data-path', default="/bigdata/eldawylab/sdas050/MS_Research/SpectralGPT_chips_multitemporal/NorthCentMN", help='dataset')
 # CA multi-temporal
 # parser.add_argument('--data-path', default="/bigdata/eldawylab/sdas050/MS_Research/SpectralGPT_chips_multitemporal/CA", help='dataset')
 # 
 parser.add_argument('--device', default='cuda', help='device')
 # ()
 parser.add_argument('--num-classes', default=12, type=int, help='num_classes')
 # GPUbatch_size
 # parser.add_argument('-b', '--batch-size', default=32, type=int,
 # help='images per gpu, the total batch size is $NGPU x batch_size')
 parser.add_argument('-b', '--batch-size', default=16, type=int,
 help='images per gpu, the total batch size is $NGPU x batch_size')
 # epoch
 parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
 parser.add_argument("--pretrain-path", default="/bigdata/eldawylab/sdas050/MS_Research/weights/SpectralGPT+.pth")
 parser.add_argument("--warmup-epochs", default=15,type=int)
 # epoch
 parser.add_argument('--epochs', default=250, type=int, metavar='N',
 help='number of total epochs to run')
 # BN(GPU)
 parser.add_argument('--sync_bn', type=bool, default=False, help='whether using SyncBatchNorm')
 # 
 parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', # CHANGED: 16 -> 8 (save memory)
 help='number of data loading workers (default: 4)')
 # 0.01(nGPUn)
 parser.add_argument('--lr', default=0.0001, type=float,
 help='initial learning rate')
 parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
 metavar='W', help='weight decay (default: 1e-4)',
 dest='weight_decay') # FIXED: typo in original
 # REMOVED: --save-best argument (not needed anymore)
 # 
 parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
 # 
 parser.add_argument('--output-dir', default='./multi_train/', help='path where to save')
 # 
 parser.add_argument('--resume', default='', help='resume from checkpoint')
 # 
 parser.add_argument(
 "--test-only",
 dest="test_only",
 help="Only test the model",
 action="store_true",
 )

 # 
 parser.add_argument('--world-size', default=1, type=int,
 help='number of distributed processes')
 parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
 parser.add_argument('--dist_on_itp', action='store_true')
 # Mixed precision training parameters
 parser.add_argument("--amp", default=True, type=bool, # CHANGED: False -> True (save memory)
 help="Use torch.cuda.amp for mixed precision training")

 args = parser.parse_args()

 # 
 if args.output_dir:
 mkdir(args.output_dir)

 main(args)


 # python -m torch.distributed.launch --nproc_per_node=4 --master_port=25643 --use_env train_multi_GPU_new.py