import os
import time
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import rasterio

# Import from local src/ (same as training script)
from src.models_vit_tensor_CD_2 import vit_base_patch8

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def time_synchronized():
 torch.cuda.synchronize() if torch.cuda.is_available() else None
 return time.time()

def open_image(img_path):
 with rasterio.open(img_path) as src:
 img = src.read()
 img = np.transpose(img, (1, 2, 0))
 return img.astype(np.float32)

def main():
 weights_path = "multi_train/best_model.pth" # Relative path
 image_folder_path = "/bigdata/eldawylab/sdas050/MS_Research/SpectralGPT_chips_multitemporal/SouthCA"
 output_dir = "/bigdata/eldawylab/sdas050/MS_Research/predictions/spectralgpt_southCA"
 os.makedirs(output_dir, exist_ok=True)
 
 num_classes = 13
 
 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 print(f"Using {device} device.")

 model = vit_base_patch8(num_classes=num_classes)
 checkpoint = torch.load(weights_path, map_location=device)
 model.load_state_dict(checkpoint['model'])
 model.to(device)
 model.eval()
 
 print(f" Loaded checkpoint\n")
 
 image_files = [f for f in os.listdir(image_folder_path) 
 if f.startswith('chip_256_') and f.endswith('.tif') and not f.endswith('.mask.tif')]
 
 print(f"Found {len(image_files)} images\n")
 
 for idx, image_file_name in enumerate(sorted(image_files)):
 print(f"[{idx+1}/{len(image_files)}] {image_file_name}")
 
 try:
 img_path = os.path.join(image_folder_path, image_file_name)
 img = open_image(img_path)
 
 img_min = img.min(axis=(0, 1), keepdims=True)
 img_max = img.max(axis=(0, 1), keepdims=True)
 img = (img - img_min) / (img_max - img_min + 1e-8)
 
 img = transforms.ToTensor()(img)
 img = torch.unsqueeze(img, dim=0).to(device)
 
 with torch.no_grad():
 t_start = time_synchronized()
 output = model(img)
 t_end = time_synchronized()
 
 prediction = output['out'].argmax(1).squeeze(0)
 prediction = prediction.cpu().numpy().astype(np.uint8)
 
 output_name = image_file_name.replace('.tif', '.npy')
 np.save(os.path.join(output_dir, output_name), prediction)
 
 output_png = image_file_name.replace('.tif', '.png')
 Image.fromarray(prediction).save(os.path.join(output_dir, output_png))
 
 print(f" {t_end - t_start:.3f}s\n")
 
 except Exception as e:
 print(f"Error: {e}\n")
 continue
 
 print("="*60)
 print(f"Complete! {output_dir}")
 print("="*60)

if __name__ == '__main__':
 main()