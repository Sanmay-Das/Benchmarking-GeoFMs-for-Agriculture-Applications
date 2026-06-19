"""
Custom pipeline transforms for SatMAE segmentation
"""
import torch
import numpy as np
from mmseg.datasets.builder import PIPELINES

@PIPELINES.register_module()
class DebugShapes:
    """Print shapes of img and gt_semantic_seg for debugging"""
    
    def __call__(self, results):
        import torch
        print(f"\n[DEBUG DebugShapes]")
        if 'img' in results:
            img = results['img']
            if isinstance(img, torch.Tensor):
                print(f"  img type: torch.Tensor, shape: {img.shape}, dtype: {img.dtype}")
            else:
                print(f"  img type: {type(img)}, shape: {img.shape if hasattr(img, 'shape') else 'N/A'}")
        
        if 'gt_semantic_seg' in results:
            mask = results['gt_semantic_seg']
            if isinstance(mask, torch.Tensor):
                print(f"  gt_semantic_seg type: torch.Tensor, shape: {mask.shape}, dtype: {mask.dtype}")
            else:
                print(f"  gt_semantic_seg type: {type(mask)}, shape: {mask.shape if hasattr(mask, 'shape') else 'N/A'}")
        
        return results



@PIPELINES.register_module()
class EnsureSegMaskShape:
    """
    Ensure segmentation mask has correct 2D shape (H, W)
    Fixes issues with flattened or incorrectly shaped masks
    """
    
    def __call__(self, results):
        if 'gt_semantic_seg' in results:
            mask = results['gt_semantic_seg']
            
            # If torch tensor, convert to numpy
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            
            # Ensure 2D shape
            if mask.ndim == 1:
                # Flattened - reshape to square
                size = int(np.sqrt(mask.shape[0]))
                mask = mask.reshape(size, size)
            elif mask.ndim == 3:
                # Has channel dimension - squeeze it
                mask = mask.squeeze()
            
            # Store back
            results['gt_semantic_seg'] = mask
            
        return results
    
    def __repr__(self):
        return self.__class__.__name__