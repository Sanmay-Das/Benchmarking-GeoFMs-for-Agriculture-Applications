# import os
# import torch
# import numpy as np
# import torch.utils.data as data
# from PIL import Image
# import skimage.io as io
# from imgaug import augmenters as iaa
# import torchvision.transforms.functional as transF


# class SegDataset(data.Dataset):
#     """
#     Dataset for SpectralGPT semantic segmentation
#     Works with multi-temporal chips (18 bands = 6 bands x 3 timesteps)
#     Uses [0, 1] Min-Max normalization with GLOBAL statistics
#     """
    
#     def __init__(self, image_root, txt_name: str = "train.txt", training=False, data_name="IowaChips"):
#         super(SegDataset, self).__init__()
        
#         assert os.path.exists(image_root), f"path '{image_root}' does not exist."
        
#         # Read chip names from txt
#         txt_path = os.path.join(image_root, "ImageSets", txt_name)
#         assert os.path.exists(txt_path), f"file '{txt_path}' does not exist."
        
#         with open(txt_path, "r") as f:
#             file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        
#         self.training = training
        
#         # ===================================================================
#         # Load exact chip names (e.g., CentIA_chip_0_0)
#         # ===================================================================
#         self.images = []
#         self.masks = []
        
#         for x in file_names:
#             img_path = os.path.join(image_root, "ImageSets", f"{x}.tif")
#             mask_path = os.path.join(image_root, "ImageSets", f"{x}_mask.tif")
            
#             # Quick safety check to warn if a file is missing
#             if not os.path.exists(img_path):
#                 print(f"Warning: Cannot find {img_path}")
#                 continue
                
#             self.images.append(img_path)
#             self.masks.append(mask_path)
        
#         assert len(self.images) == len(self.masks), "Mismatch between images and masks!"
#         print(f"Loaded {len(self.images)} samples from {txt_name}")
        
#         # ===================================================================
#         # SpectralGPT [0, 1] Min-Max Normalization with GLOBAL stats
#         # ===================================================================
#         # Method: Min-Max scaling to [0, 1] range
#         # Formula: normalized = (x - global_min) / (global_max - global_min)
#         # Reference: SpectralGPT paper Section 2.8, Table 5(d)
        
#         # NOTE: Update these with the outputs from your Iowa normalization script
#         # IA Crop Segmentation
#         self.global_min = np.array([
#             296.00, 1090.00, 798.00, 930.00, 1092.00, 1079.00,  # T1
#             924.00, 1085.00, 892.00, 1009.00, 1072.00, 1050.00,  # T2
#             289.00, 748.00, 844.00, 985.00, 1031.00, 1022.00   # T3
#         ], dtype=np.float32).reshape(18, 1, 1)
        
#         self.global_max = np.array([
#             12539.00, 12638.00, 14299.00, 10808.00, 13691.00, 15615.00,  # T1
#             17267.00, 16242.00, 15310.00, 10095.00, 12202.00, 14097.00,  # T2
#             18592.00, 17680.00, 17056.00, 16474.00, 16108.00, 16053.00   # T3
#         ], dtype=np.float32).reshape(18, 1, 1)
        
#         # Calculate range (max - min)
#         self.global_range = self.global_max - self.global_min
#         # ===================================================================
        
#         # Data augmentation
#         self.transform = iaa.Sequential([
#             iaa.Rot90([0, 1, 2, 3]),
#             iaa.VerticalFlip(p=0.5),
#             iaa.HorizontalFlip(p=0.5),
#         ])
    
#     def __getitem__(self, index):
#         """
#         Returns:
#             img: [18, 128, 128] tensor normalized to [0, 1]
#             target: [128, 128] tensor with class labels
#         """
#         # Load image
#         img = open_image(self.images[index])  # (128, 128, 18)
        
#         # Load mask
#         target = np.array(Image.open(self.masks[index]).convert("P"))
        
#         # Augmentation
#         if self.training:
#             img, target = self.transform(
#                 image=img, 
#                 segmentation_maps=np.stack(
#                     (target[np.newaxis, :, :], target[np.newaxis, :, :]), 
#                     axis=-1
#                 )
#             )
#             target = target[0, :, :, 0]
        
#         # Convert to tensor
#         img = torch.tensor(img.copy(), dtype=torch.float32).permute(2, 0, 1)  # [18, 128, 128]
#         target = torch.tensor(target.copy(), dtype=torch.long)  # [128, 128]

#         # ===================================================================
#         # SpectralGPT [0, 1] normalization using GLOBAL statistics
#         # ===================================================================
#         img = (img - torch.from_numpy(self.global_min)) / torch.from_numpy(self.global_range)
#         # ===================================================================
        
#         return img, target
    
#     def __len__(self):
#         return len(self.images)
    
#     @staticmethod
#     def collate_fn(batch):
#         images, targets = list(zip(*batch))
#         batched_imgs = cat_list(images, fill_value=0)
#         batched_targets = cat_list(targets, fill_value=255)
#         return batched_imgs, batched_targets


# def cat_list(images, fill_value=0):
#     max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
#     batch_shape = (len(images),) + max_size
#     batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
#     for img, pad_img in zip(images, batched_imgs):
#         pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
#     return batched_imgs


# def open_image(img_path):
#     """
#     Load multi-temporal image
#     Handles any number of bands (10, 18, 24, etc.)
#     """    
#     img = io.imread(img_path)
    
#     # Handle different orientations
#     if img.ndim == 3:
#         # Check if bands are in first dimension
#         if img.shape[0] in [10, 18, 24]:  # Common band counts
#             img = img.transpose(1, 2, 0)  # (bands, H, W) -> (H, W, bands)
#         # If shape is already (H, W, bands), keep as is
    
#     return img.astype(np.float32)

import os

import torch

import numpy as np

import torch.utils.data as data

from PIL import Image

import skimage.io as io

from imgaug import augmenters as iaa

import torchvision.transforms.functional as transF





class SegDataset(data.Dataset):

    """

    Dataset for SpectralGPT semantic segmentation

    Works with multi-temporal chips (18 bands = 6 bands × 3 timesteps)

    Uses [0, 1] Min-Max normalization with GLOBAL statistics

    """

   

    def __init__(self, image_root, txt_name: str = "train.txt", training=False, data_name="MinnesotaChips"):

        super(SegDataset, self).__init__()

       

        assert os.path.exists(image_root), f"path '{image_root}' does not exist."

       

        # Read chip names from txt

        txt_path = os.path.join(image_root, "ImageSets", txt_name)

        assert os.path.exists(txt_path), f"file '{txt_path}' does not exist."

       

        with open(txt_path, "r") as f:

            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

       

        self.training = training

       

        # ═══════════════════════════════════════════════════════════════════

        # ✅ FIXED: Handle relative paths (../NorthCA/chip_X)

        # ═══════════════════════════════════════════════════════════════════

        self.images = []

        self.masks = []

       

        for x in file_names:

            if x.startswith('../'):

                # Relative path from image_root (e.g., ../NorthCA/chip_X)

                img_path = os.path.join(image_root, x + ".tif")

                mask_path = os.path.join(image_root, x + "_mask.tif")

            else:

                # Simple chip name (e.g., chip_X) - look in ImageSets folder

                img_path = os.path.join(image_root, "ImageSets", f"{x}.tif")

                mask_path = os.path.join(image_root, "ImageSets", f"{x}_mask.tif")

           

            self.images.append(img_path)

            self.masks.append(mask_path)

       

        assert len(self.images) == len(self.masks)

        print(f"Loaded {len(self.images)} samples from {txt_name}")

       

        # ═══════════════════════════════════════════════════════════════════

        # ✅ ADDED: SpectralGPT [0, 1] Min-Max Normalization with GLOBAL stats

        # ═══════════════════════════════════════════════════════════════════

        # Calculated from NorthCA training set (7,937 chips)

        # Method: Min-Max scaling to [0, 1] range

        # Formula: normalized = (x - global_min) / (global_max - global_min)

        # Reference: SpectralGPT paper Section 2.8, Table 5(d)

        # IA
        # self.global_min = np.array([

        #     296.00, 1090.00, 798.00, 929.00, 1079.00, 1079.00,  # T1: B02, B03, B04, B8A, B11, B12

        #     525.00, 1076.00, 892.00, 1009.00, 1072.00, 1028.00,  # T2: B02, B03, B04, B8A, B11, B12

        #     289.00, 748.00, 844.00, 978.00, 1031.00, 1022.00   # T3: B02, B03, B04, B8A, B11, B12

        # ], dtype=np.float32).reshape(18, 1, 1)

       

        # self.global_max = np.array([

        #     12539.00, 12638.00, 14299.00, 10808.00, 15296.00, 16058.00,  # T1

        #     17267.00, 16242.00, 16593.00, 11714.00, 12628.00, 14097.00,  # T2

        #     18592.00, 17680.00, 17056.00, 16474.00, 16108.00, 16053.00   # T3

        # ], dtype=np.float32).reshape(18, 1, 1)



        # CA
        # self.global_min = np.array([

        #     347.00, 958.00, 783.00, 896.00, 997.00, 994.00,    # T1: B02, B03, B04, B8A, B11, B12
        #     811.00, 934.00, 936.00, 506.00, 999.00, 991.00,    # T2: B02, B03, B04, B8A, B11, B12
        #     392.00, 854.00, 916.00, 934.00, 997.00, 993.00     # T3: B02, B03, B04, B8A, B11, B12
        # ], dtype=np.float32).reshape(18, 1, 1)

       

        # self.global_max = np.array([

        #     18912.00, 17984.00, 16932.00, 15344.00, 15434.00, 16073.00,  # T1
        #     18877.00, 17452.00, 17009.00, 15409.00, 15174.00, 16092.00,  # T2
        #     19216.00, 17262.00, 17256.00, 16531.00, 16179.00, 16106.00   # T3
        # ], dtype=np.float32).reshape(18, 1, 1)


        # IL
        # self.global_min = np.array([
        #     675.00, 1012.00, 1052.00, 984.00, 1031.00, 1005.00,    # T1: B02, B03, B04, B8A, B11, B12
        #     872.00,  694.00,  785.00, 1047.00, 1116.00, 1099.00,   # T2: B02, B03, B04, B8A, B11, B12
        #     517.00,  787.00,  829.00, 1040.00, 1013.00,  996.00    # T3: B02, B03, B04, B8A, B11, B12
        # ], dtype=np.float32).reshape(18, 1, 1)

        # self.global_max = np.array([
        #     18688.00, 17872.00, 17231.00, 16633.00, 15868.00, 16061.00,  # T1
        #     12834.00, 13445.00, 14211.00, 10290.00, 13408.00, 14751.00,  # T2
        #     18131.00, 17536.00, 16976.00, 13643.00, 16100.00, 16052.00   # T3
        # ], dtype=np.float32).reshape(18, 1, 1)

        # NC
        # self.global_min = np.array([
        #     0.00, 0.00, 0.00, 0.00, 0.00, 0.00,    # T1: B02, B03, B04, B8A, B11, B12
        #     0.00, 0.00, 0.00, 0.00, 0.00, 0.00,    # T2: B02, B03, B04, B8A, B11, B12
        #     0.00, 0.00, 0.00, 0.00, 0.00, 0.00     # T3: B02, B03, B04, B8A, B11, B12
        # ], dtype=np.float32).reshape(18, 1, 1)

        # self.global_max = np.array([
        #     15325.00, 15699.00, 16115.00, 13628.00, 14242.00, 15093.00,  # T1
        #     17100.00, 16488.00, 16918.00, 11342.00, 14423.00, 16374.00,  # T2
        #     15930.00, 15386.00, 15342.00, 11933.00, 14433.00, 16067.00   # T3
        # ], dtype=np.float32).reshape(18, 1, 1)

        # MN  
        self.global_min = np.array([
            641.00, 1018.00, 1013.00,  921.00, 1030.00, 1022.00,   # T1: B02, B03, B04, B8A, B11, B12
            764.00,  918.00,  675.00,  713.00,  992.00,  991.00,   # T2: B02, B03, B04, B8A, B11, B12
            687.00, 1000.00,  977.00,  881.00, 1053.00, 1030.00    # T3: B02, B03, B04, B8A, B11, B12
        ], dtype=np.float32).reshape(18, 1, 1)

        self.global_max = np.array([
            19467.00, 18496.00, 17621.00, 13263.00, 15384.00, 16124.00,  # T1: B02, B03, B04, B8A, B11, B12
            18711.00, 18035.00, 17259.00, 10610.00, 12480.00, 14638.00,  # T2: B02, B03, B04, B8A, B11, B12
            18656.00, 17712.00, 17120.00, 15848.00, 16108.00, 16053.00   # T3: B02, B03, B04, B8A, B11, B12
        ], dtype=np.float32).reshape(18, 1, 1)      

        # Calculate range (max - min)

        self.global_range = self.global_max - self.global_min

        # ═══════════════════════════════════════════════════════════════════

       

        # Data augmentation

        self.transform = iaa.Sequential([

            iaa.Rot90([0, 1, 2, 3]),

            iaa.VerticalFlip(p=0.5),

            iaa.HorizontalFlip(p=0.5),

        ])

   

    def __getitem__(self, index):

        """

        Returns:

            img: [18, 128, 128] tensor normalized to [0, 1]

            target: [128, 128] tensor with class labels

        """

        # Load image

        img = open_image(self.images[index])  # (128, 128, 18)

       

        # Load mask

        target = np.array(Image.open(self.masks[index]).convert("P"))
        target[target == 0] = 255       # No Data -> ignore index
        target[target != 255] -= 1      # 1-13 -> 0-12
       

        # Augmentation

        if self.training:

            img, target = self.transform(

                image=img,

                segmentation_maps=np.stack(

                    (target[np.newaxis, :, :], target[np.newaxis, :, :]),

                    axis=-1

                )

            )

            target = target[0, :, :, 0]

       

        # Convert to tensor

        img = torch.tensor(img.copy(), dtype=torch.float32).permute(2, 0, 1)  # [18, 128, 128]

        target = torch.tensor(target.copy(), dtype=torch.long)  # [128, 128]



        # ═══════════════════════════════════════════════════════════════════

        # ✅ REPLACED: Global normalization (not per-image!)

        # ═══════════════════════════════════════════════════════════════════

        # SpectralGPT [0, 1] normalization using GLOBAL statistics

        img = (img - torch.from_numpy(self.global_min)) / torch.from_numpy(self.global_range)
        img = img.clamp(0.0, 1.0)

        # ═══════════════════════════════════════════════════════════════════

       

        return img, target

   

    def __len__(self):

        return len(self.images)

   

    @staticmethod

    def collate_fn(batch):

        images, targets = list(zip(*batch))

        batched_imgs = cat_list(images, fill_value=0)

        batched_targets = cat_list(targets, fill_value=255)

        return batched_imgs, batched_targets





def cat_list(images, fill_value=0):

    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))

    batch_shape = (len(images),) + max_size

    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)

    for img, pad_img in zip(images, batched_imgs):

        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)

    return batched_imgs





def open_image(img_path):

    """

    Load multi-temporal image

    Handles any number of bands (10, 18, 24, etc.)

    """    

    img = io.imread(img_path)

   

    # Handle different orientations

    if img.ndim == 3:

        # Check if bands are in first dimension

        if img.shape[0] in [10, 18, 24]:  # Common band counts

            img = img.transpose(1, 2, 0)  # (bands, H, W) → (H, W, bands)

        # If shape is already (H, W, bands), keep as is

   

    return img.astype(np.float32)