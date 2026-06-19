import os
import numpy as np
import torch
import glob
import sys
from PIL import Image

sys.path.insert(0, '/bigdata/eldawylab/sdas050/MS_Research/IEEE_TPAMI_SpectralGPT/downstream_tasks/SegMunich')
import train_utils.distributed_utils as utils

CLASS_NAMES = [
    "Natural Vegetation", "Forest", "Corn", "Soybeans", "Wetlands",
    "Developed/Barren", "Open Water", "Winter Wheat", "Alfalfa",
    "Fallow/Idle Cropland", "Cotton", "Sorghum", "Other"
]

def print_table(confmat):
    acc_global, acc_r, acc_p, iu = confmat.compute()

    print(f"\n+----------------------+-------+-------+")
    print(f"|        Class         |  IoU  |  Acc  |")
    print(f"+----------------------+-------+-------+")
    for i, name in enumerate(CLASS_NAMES):
        iou = iu[i].item() * 100
        acc = acc_r[i].item() * 100
        iou_str = f"{iou:.2f}" if not np.isnan(iou) else "nan"
        acc_str = f"{acc:.2f}" if not np.isnan(acc) else "nan"
        print(f"| {name:<20} | {iou_str:>5} | {acc_str:>5} |")
    print(f"+----------------------+-------+-------+")
    print(f"\n+-------+-------+-------+")
    print(f"|  aAcc |  mIoU |  mAcc |")
    print(f"+-------+-------+-------+")
    print(f"| {acc_global.item()*100:.2f} | {iu.nanmean().item()*100:.2f} | {acc_r.nanmean().item()*100:.2f} |")
    print(f"+-------+-------+-------+")

def main():
    pred_dir = '/bigdata/eldawylab/sdas050/MS_Research/predictions/prithvi_EastNC'
    gt_dir   = '/bigdata/eldawylab/sdas050/MS_Research/data/multi_temporal_crop_segmentation/EastNC'
    num_classes = 13

    pred_files = sorted(glob.glob(os.path.join(pred_dir, 'chip_*.npy')))
    print(f"Found {len(pred_files)} prediction files\n")

    confmat = utils.ConfusionMatrix(num_classes)

    for pred_path in pred_files:
        chip_name = os.path.basename(pred_path).replace('.npy', '')
        mask_path = os.path.join(gt_dir, f'{chip_name}.mask.tif')

        if not os.path.exists(mask_path):
            print(f"⚠️  Mask not found: {chip_name}")
            continue

        pred = np.load(pred_path)

        gt = np.array(Image.open(mask_path))
        gt = gt.astype(np.int32)
        gt[gt == 0] = 255
        gt[gt != 255] -= 1

        pred_tensor = torch.tensor(pred.flatten(), dtype=torch.long)
        gt_tensor   = torch.tensor(gt.flatten(),   dtype=torch.long)
        confmat.update(gt_tensor, pred_tensor)

    print_table(confmat)

if __name__ == '__main__':
    main()