"""
SatMAE Backbone for MMSegmentation 0.30.0
Wraps SatMAE's ViT-Large encoder for use with MMSeg decode heads
"""

import torch
import torch.nn as nn
from mmcv.runner import BaseModule  
from mmseg.models.builder import BACKBONES

from models_vit_group_channels import GroupChannelsVisionTransformer


@BACKBONES.register_module()
class SatMAEBackbone(BaseModule):
    """
    SatMAE ViT-Large backbone for MMSegmentation
    
    Args:
        pretrained (str): Path to pretrained weights
        img_size (int): Input image size (default: 96)
        patch_size (int): Patch size (default: 8)
        in_chans (int): Number of input channels (default: 18)
        embed_dim (int): Embedding dimension (default: 1024)
        depth (int): Number of transformer blocks (default: 24)
        num_heads (int): Number of attention heads (default: 16)
        channel_groups (list): Band grouping for multi-spectral data
        drop_path_rate (float): Stochastic depth rate (default: 0.2)
    """
    
    def __init__(self,
                 pretrained=None,
                 img_size=96,
                 patch_size=8,
                 in_chans=18,
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 channel_groups=None,
                 drop_path_rate=0.2,
                 init_cfg=None):
        super(SatMAEBackbone, self).__init__(init_cfg)
        
        if channel_groups is None:
            channel_groups = [[0, 1, 2, 6], [3, 4, 5, 7], [8, 9]]
        
        # Create SatMAE ViT encoder
        self.encoder = GroupChannelsVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,  # No classification head
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            channel_groups=channel_groups,
            drop_path_rate=drop_path_rate,
            global_pool=False,
        )
        
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        
        # Load pretrained weights if provided
        if pretrained is not None:
            self.load_pretrained(pretrained)
    
    def load_pretrained(self, pretrained_path):
        """Load pretrained weights from SatMAE checkpoint"""
        print(f"Loading SatMAE pretrained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remove decoder keys (MAE decoder not needed for segmentation)
        encoder_dict = {}
        for k, v in state_dict.items():
            if not k.startswith('decoder') and not k.startswith('mask_token'):
                # Remove classification head keys
                if not k.startswith('head') and not k.startswith('fc_norm') and not k.startswith('channel_cls'):
                    encoder_dict[k] = v
        
        # Load weights
        msg = self.encoder.load_state_dict(encoder_dict, strict=False)
        print(f"Loaded pretrained weights.")
        print(f"Missing keys: {len(msg.missing_keys)} (expected for segmentation)")
        print(f"Unexpected keys: {len(msg.unexpected_keys)}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (B, C, H, W) input tensor
        
        Returns:
            list: Feature maps for segmentation decoder
        """
        # Get features from encoder
        features = self.encoder.forward_features(x)
        
        # Handle different output formats from encoder
        if features.dim() == 2:
            # If encoder returns (B, C) - global pooled features
            B, C = features.shape
            
            # Calculate spatial dimensions: 96 // 8 = 12
            H_out = W_out = self.img_size // self.patch_size
            
            # Reshape to spatial format: (B, C) -> (B, C, H, W)
            features = features.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            features = features.expand(B, C, H_out, W_out)  # (B, C, 12, 12)
            
        elif features.dim() == 3:
            # If encoder returns (B, N, C) - patch tokens
            B, N, C = features.shape
            H = W = int(N ** 0.5)  # sqrt(144) = 12
            features = features.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")
        
        # MMSeg expects a list of feature maps
        return [features]


    
    def init_weights(self, pretrained=None):
        """Initialize weights (handled by load_pretrained)"""
        pass