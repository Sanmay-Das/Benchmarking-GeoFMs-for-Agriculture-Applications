import os
import numpy as np
import torch
import glob
import sys

sys.path.insert(0, '/bigdata/eldawylab/sdas050/MS_Research/IEEE_TPAMI_SpectralGPT/downstream_tasks/SegMunich')
import train_utils.distributed_utils as utils
from PIL import Image

def main():
    pred_dir = '/bigdata/eldawylab/sdas050/MS_Research/predictions/spectralgpt_EastNC'
    gt_dir   = '/bigdata/eldawylab/sdas050/MS_Research/SpectralGPT_chips_multitemporal/EastNC/ImageSets'
    num_classes = 13

    pred_files = sorted(glob.glob(os.path.join(pred_dir, 'EastNC_chip_*.npy')))
    print(f"Found {len(pred_files)} prediction files\n")

    confmat = utils.ConfusionMatrix(num_classes)

    for pred_path in pred_files:
        chip_name = os.path.basename(pred_path).replace('.npy', '')  # EastNC_chip_320_64
        mask_path = os.path.join(gt_dir, f'{chip_name}_mask.tif')

        if not os.path.exists(mask_path):
            print(f"⚠️  Mask not found: {chip_name}")
            continue

        # Load prediction (0-12)
        pred = np.load(pred_path)

        # Load GT (1-13, 0=NoData)
        from PIL import Image
        gt = np.array(Image.open(mask_path))
        gt = gt.astype(np.int32)
        gt[gt == 0] = 255        # NoData -> ignore
        gt[gt != 255] -= 1       # 1-13 -> 0-12

        # Update confusion matrix
        pred_tensor = torch.tensor(pred.flatten(), dtype=torch.long)
        gt_tensor   = torch.tensor(gt.flatten(),   dtype=torch.long)
        confmat.update(gt_tensor, pred_tensor)

    # Print results
    print(str(confmat))

if __name__ == '__main__':
    main()