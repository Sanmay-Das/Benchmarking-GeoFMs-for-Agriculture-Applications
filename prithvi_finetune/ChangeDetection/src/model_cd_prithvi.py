"""
model_cd_prithvi.py
-------------------
Prithvi Change Detection model — standalone PyTorch, no mmseg dependencies.

Architecture:
    Shared TemporalViTEncoder (Prithvi pretrained)
        T1 (B,6,224,224) -> encoder -> tokens (B, 196, 768) @ 14x14
        T2 (B,6,224,224) -> encoder -> tokens (B, 196, 768) @ 14x14
                (shared weights — Siamese)
        diff = f1 - f2               (B, 768, 14, 14)
        FPN conv pyramid + FPNHEAD -> (B, 256, 224, 224)
        cls_seg                    -> (B, 2,   224, 224)
        LogSoftmax                 -> binary change map

FPN conv pyramid (all branches from same diff at 14x14):
    conv0: 768->256,  16x ConvTranspose -> (B,  256, 224, 224)
    conv1: 768->512,   8x ConvTranspose -> (B,  512, 112, 112)
    conv2: 768->1024,  4x ConvTranspose -> (B, 1024,  56,  56)
    conv3: 768->2048,  2x ConvTranspose -> (B, 2048,  28,  28)
    FPNHEAD(channels=2048) fuses the 4 scales -> (B, 256, 224, 224)

This mirrors the SpectralGPT CD decoder for fair benchmarking.
Neck (ConvTransformerNeck) is bypassed — diff is computed at token level.

Place this file at:
    prithvi_finetune/ChangeDetection/src/model_cd_prithvi.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple


# ============================================================================
# Positional embedding helpers (copied from geospatial_fm.py, no mmseg)
# ============================================================================

def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos   = pos.reshape(-1)
    out   = np.einsum('m,d->md', pos, omega)
    emb   = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return emb


def get_3d_sincos_pos_embed(embed_dim: int, grid_size: tuple,
                             cls_token: bool = False):
    t_size, h_size, w_size = grid_size
    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

    w_pos = np.tile(w_pos, (t_size * h_size, 1))
    h_pos = np.tile(np.repeat(h_pos, w_size, axis=0), (t_size, 1))
    t_pos = np.repeat(t_pos, h_size * w_size, axis=0)

    pos_embed = np.concatenate((w_pos, h_pos, t_pos), axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


# ============================================================================
# Patch Embedding (copied from geospatial_fm.py, no mmseg)
# ============================================================================

class PatchEmbed3D(nn.Module):
    """3D patch embedding for Prithvi (B, C, T, H, W) -> (B, L, D)."""

    def __init__(self, img_size=224, patch_size=16, num_frames=1,
                 tubelet_size=1, in_chans=6, embed_dim=768):
        super().__init__()
        img_size   = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.grid_size = (
            num_frames // tubelet_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = (self.grid_size[0] *
                            self.grid_size[1] *
                            self.grid_size[2])
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1]),
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        x    = self.proj(x)          # (B, D, T', H', W')
        Hp   = x.shape[3]
        Wp   = x.shape[4]
        x    = x.flatten(2).transpose(1, 2)   # (B, L, D)
        return x, Hp, Wp


# ============================================================================
# TemporalViTEncoder (standalone, no mmseg @BACKBONES decorator)
# ============================================================================

class TemporalViTEncoder(nn.Module):
    """
    Prithvi ViT encoder — standalone version without mmseg dependencies.
    Identical forward pass to geospatial_fm.TemporalViTEncoder.
    """

    def __init__(self, img_size=224, patch_size=16, num_frames=1,
                 tubelet_size=1, in_chans=6, embed_dim=768,
                 depth=6, num_heads=8, mlp_ratio=4.0,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.embed_dim  = embed_dim
        self.num_frames = num_frames

        self.patch_embed = PatchEmbed3D(
            img_size, patch_size, num_frames, tubelet_size, in_chans, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self._init_weights()

    def _init_weights(self):
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.patch_embed.grid_size,
            cls_token=True,
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """
        Returns spatial token map without CLS token.
        x : (B, C, T, H, W)
        Returns: (B, embed_dim, H//patch_size, W//patch_size)
                 e.g. (B, 768, 14, 14) for 224x224 input, patch=16
        """
        x, Hp, Wp = self.patch_embed(x)         # (B, L, D)
        x = x + self.pos_embed[:, 1:, :]

        cls_token  = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)   # (B, L+1, D)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # drop CLS → spatial tokens only
        x = x[:, 1:, :]                          # (B, L, D)
        B, L, D = x.shape
        x = x.permute(0, 2, 1).reshape(B, D, Hp, Wp)   # (B, D, 14, 14)
        return x


# ============================================================================
# FPN Decoder — identical to SpectralGPT model_cd_spectralgpt.py
# ============================================================================

class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super().__init__()
        for pool_size in pool_sizes:
            self.append(nn.Sequential(
                nn.AdaptiveMaxPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            ))

    def forward(self, x):
        out = []
        for ppm in self:
            out.append(F.interpolate(ppm(x), size=x.shape[2:],
                                     mode='bilinear', align_corners=True))
        return out


class PPMHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.psp_modules = PPM(pool_sizes, in_channels, out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * out_channels,
                      out_channels, kernel_size=1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, dim=1)
        return self.final(out)


class FPNHEAD(nn.Module):
    def __init__(self, channels=2048, out_channels=256):
        super().__init__()
        self.PPMHead    = PPMHEAD(channels, out_channels)
        self.Conv_fuse1  = nn.Sequential(nn.Conv2d(channels // 2, out_channels, 1),
                                          nn.GroupNorm(16, out_channels), nn.GELU(), nn.Dropout(0.5))
        self.Conv_fuse1_ = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1),
                                          nn.GroupNorm(16, out_channels), nn.GELU(), nn.Dropout(0.5))
        self.Conv_fuse2  = nn.Sequential(nn.Conv2d(channels // 4, out_channels, 1),
                                          nn.GroupNorm(16, out_channels), nn.GELU(), nn.Dropout(0.5))
        self.Conv_fuse2_ = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1),
                                          nn.GroupNorm(16, out_channels), nn.GELU(), nn.Dropout(0.5))
        self.Conv_fuse3  = nn.Sequential(nn.Conv2d(channels // 8, out_channels, 1),
                                          nn.GroupNorm(16, out_channels), nn.GELU(), nn.Dropout(0.5))
        self.Conv_fuse3_ = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1),
                                          nn.GroupNorm(16, out_channels), nn.GELU(), nn.Dropout(0.5))
        self.fuse_all    = nn.Sequential(nn.Conv2d(out_channels * 4, out_channels, 1),
                                          nn.GroupNorm(16, out_channels), nn.GELU(), nn.Dropout(0.5))
        self.conv_x1     = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, fpn):
        x1 = self.PPMHead(fpn[-1])
        x  = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x2 = self.Conv_fuse1_(self.conv_x1(x) + self.Conv_fuse1(fpn[-2]))
        x  = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        x3 = self.Conv_fuse2_(x + self.Conv_fuse2(fpn[-3]))
        x  = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x4 = self.Conv_fuse3_(x + self.Conv_fuse3(fpn[-4]))

        x1 = F.interpolate(x1, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x4.shape[2:], mode='bilinear', align_corners=True)
        return self.fuse_all(torch.cat([x1, x2, x3, x4], dim=1))


# ============================================================================
# Prithvi CD Model
# ============================================================================

class Prithvi_CD(nn.Module):
    """
    Siamese Prithvi encoder + FPN change detection head.

    Forward pass:
        t1, t2 : (B, 6, 224, 224)
            -> unsqueeze T dim -> (B, 6, 1, 224, 224)
            -> shared TemporalViTEncoder.forward_features()
            -> f1, f2 : (B, 768, 14, 14)
            -> diff = f1 - f2
            -> FPN conv pyramid + FPNHEAD -> (B, 256, 224, 224)
            -> cls_seg                    -> (B, 2,   224, 224) log-probs
    """

    def __init__(self):
        super().__init__()

        # shared encoder
        self.encoder = TemporalViTEncoder(
            img_size=224,
            patch_size=16,
            num_frames=1,
            tubelet_size=1,
            in_chans=6,
            embed_dim=768,
            depth=6,
            num_heads=8,
            mlp_ratio=4.0,
        )

        # FPN conv pyramid: all branches from diff (B, 768, 14, 14)
        # 14 * 16 = 224, 14 * 8 = 112, 14 * 4 = 56, 14 * 2 = 28
        self.conv0 = nn.Sequential(
            nn.Conv2d(768, 512, 1),
            nn.GroupNorm(32, 512), nn.GELU(),
            nn.ConvTranspose2d(512, 256, 16, 16),   # (B, 256, 224, 224)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(768, 512, 1),
            nn.GroupNorm(32, 512), nn.GELU(),
            nn.ConvTranspose2d(512, 512, 8, 8),     # (B, 512, 112, 112)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(768, 1024, 1),
            nn.GroupNorm(32, 1024), nn.GELU(),
            nn.ConvTranspose2d(1024, 1024, 4, 4),   # (B, 1024, 56, 56)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(768, 2048, 1),
            nn.GroupNorm(32, 2048), nn.GELU(),
            nn.ConvTranspose2d(2048, 2048, 2, 2),   # (B, 2048, 28, 28)
        )

        # FPN decoder + binary classification head
        self.decoder = FPNHEAD(channels=2048, out_channels=256)
        self.cls_seg  = nn.Conv2d(256, 2, kernel_size=3, padding=1)
        self.sm       = nn.LogSoftmax(dim=1)

    def _encode(self, x):
        """
        x : (B, 6, 224, 224)
        returns feature map : (B, 768, 14, 14)
        """
        x = x.unsqueeze(2)                            # (B, 6, 1, 224, 224)
        return self.encoder.forward_features(x)       # (B, 768, 14, 14)

    def forward(self, t1, t2):
        """
        t1, t2  : (B, 6, 224, 224)
        returns : (B, 2, 224, 224) log-probabilities
        """
        f1   = self._encode(t1)     # (B, 768, 14, 14)
        f2   = self._encode(t2)     # (B, 768, 14, 14)
        diff = f1 - f2              # element-wise difference

        # multi-scale pyramid
        m = [
            self.conv0(diff),   # (B,  256, 224, 224)
            self.conv1(diff),   # (B,  512, 112, 112)
            self.conv2(diff),   # (B, 1024,  56,  56)
            self.conv3(diff),   # (B, 2048,  28,  28)
        ]

        x = self.decoder(m)     # (B, 256, 224, 224)
        x = self.cls_seg(x)     # (B, 2,   224, 224)
        return self.sm(x)


# ============================================================================
# Constructor + weight loading
# ============================================================================

def build_prithvi_cd(pretrain_path: str = None) -> Prithvi_CD:
    """
    Build Prithvi CD model and optionally load pretrained encoder weights.

    Args:
        pretrain_path : path to Prithvi_EO_V1_100M.pt
    Returns:
        Prithvi_CD instance
    """
    model = Prithvi_CD()

    if pretrain_path is not None:
        print(f"Load pre-trained checkpoint from: {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location='cpu')

        # Prithvi checkpoint may be flat dict or nested under 'model'
        if 'model' in checkpoint:
            ckpt = checkpoint['model']
        else:
            ckpt = checkpoint

        # Add 'encoder.' prefix to match our model's attribute name
        # Prithvi checkpoint keys are like: patch_embed.proj.weight
        # Our model keys are like:         encoder.patch_embed.proj.weight
        ckpt_prefixed = {}
        for k, v in ckpt.items():
            # Skip decoder keys (not needed for CD)
            if any(k.startswith(p) for p in
                   ['decoder', 'mask_token', 'decoder_pred',
                    'decoder_embed', 'decoder_norm']):
                continue
            ckpt_prefixed[f'encoder.{k}'] = v

        msg = model.load_state_dict(ckpt_prefixed, strict=False)
        print(msg)

    return model


if __name__ == '__main__':
    # Quick shape test
    # Run from ChangeDetection/ folder:
    # python src/model_cd_prithvi.py
    model = build_prithvi_cd(pretrain_path=None)
    t1  = torch.rand(2, 6, 224, 224)
    t2  = torch.rand(2, 6, 224, 224)
    out = model(t1, t2)
    print(f"Output shape: {out.shape}")   # expected: (2, 2, 224, 224)
    assert out.shape == (2, 2, 224, 224), "Shape mismatch!"
    print("Shape test passed.")
