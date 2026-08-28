"""
model_cd_satmae.py
------------------
SatMAE Change Detection model for benchmark.

Architecture:
    Shared GroupChannelsVisionTransformer encoder (SatMAE pretrained)
        T1 (B, 6, 96, 96) -> forward_features_seg() -> f1 (B, 1024, 12, 12)
        T2 (B, 6, 96, 96) -> forward_features_seg() -> f2 (B, 1024, 12, 12)
                (shared weights -- Siamese)
        diff = f1 - f2               (B, 1024, 12, 12)
        FPN conv pyramid + FPNHEAD -> (B, 256, 96, 96)
        cls_seg                    -> (B, 2,   96, 96)
        LogSoftmax                 -> binary change map

FPN conv pyramid (all branches from same diff):
    conv0: 1024->256,  8x ConvTranspose -> (B,  256, 96, 96)
    conv1: 1024->512,  4x ConvTranspose -> (B,  512, 48, 48)
    conv2: 1024->1024, 2x ConvTranspose -> (B, 1024, 24, 24)
    conv3: 1024->2048, no upsample      -> (B, 2048, 12, 12)
    FPNHEAD(channels=2048) fuses the 4 scales -> (B, 256, 96, 96)

This mirrors the SpectralGPT CD decoder (model_cd_spectralgpt.py) for fair benchmarking.

Key adaptation from segmentation:
    Original channel groups: (0,1,2,6), (3,4,5,7), (8,9) -- needs 10 bands
    CD channel groups:       (0,1,2),   (3,4,5)           -- 6 bands only
    2 groups x 3 bands = 6 bands total, matching your CD chips

    Token count: L = (96/8)^2 = 144 spatial tokens
    Feature map: 12x12 after encoder -> 8x upsample -> 96x96

Place this file at:
    SatMAE/ChangeDetection/src/model_cd_satmae.py
"""

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed

# These come from SatMAE/util/ -- copy util/ into ChangeDetection/
from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid


# ============================================================================
# GroupChannelsVisionTransformer adapted for 6-band CD input
# Copied from models_vit_group_channels.py with:
#   1. channel_groups adapted to 6 bands: ((0,1,2), (3,4,5))
#   2. forward_features_seg() kept as-is
#   3. No mmseg or torchmetrics dependencies
# ============================================================================

class GroupChannelsViTCD(timm.models.vision_transformer.VisionTransformer):
    """
    SatMAE GroupChannels ViT adapted for 6-band CD input.
    Channel groups: (0,1,2) and (3,4,5) -- 2 groups of 3 bands each.
    """

    def __init__(self, channel_embed=256,
                 channel_groups=((0, 1, 2), (3, 4, 5)),
                 **kwargs):
        super().__init__(**kwargs)

        img_size  = kwargs['img_size']
        patch_size = kwargs['patch_size']
        embed_dim  = kwargs['embed_dim']

        self.channel_groups = channel_groups

        # Replace single PatchEmbed with one per channel group
        self.patch_embed = nn.ModuleList([
            PatchEmbed(img_size, patch_size, len(group), embed_dim)
            for group in channel_groups
        ])
        num_patches = self.patch_embed[0].num_patches  # 144 for 96x96, patch=8

        # Positional embedding (spatial only, no channel dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim - channel_embed))
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Channel embedding
        num_groups = len(channel_groups)
        self.channel_embed = nn.Parameter(
            torch.zeros(1, num_groups, channel_embed))
        chan_embed = get_1d_sincos_pos_embed_from_grid(
            self.channel_embed.shape[-1],
            torch.arange(num_groups).numpy())
        self.channel_embed.data.copy_(
            torch.from_numpy(chan_embed).float().unsqueeze(0))

        # Extra CLS channel embedding
        self.channel_cls_embed = nn.Parameter(torch.zeros(1, 1, channel_embed))

    def forward_features_seg(self, x):
        """
        Returns spatial feature map for segmentation/CD.
        x : (B, 6, H, W)
        Returns: (B, embed_dim, H//patch_size, W//patch_size)
                 e.g. (B, 1024, 12, 12) for 96x96 input, patch=8
        """
        b, c, h, w = x.shape
        G = len(self.channel_groups)

        # Per-group patch embedding
        x_c_embed = []
        for i, group in enumerate(self.channel_groups):
            x_c = x[:, group, :, :]
            x_c_embed.append(self.patch_embed[i](x_c))   # (B, L, D)

        tokens = torch.stack(x_c_embed, dim=1)  # (B, G, L, D)
        _, G, L, D = tokens.shape

        # Add channel + positional embeddings
        channel_embed = self.channel_embed.unsqueeze(2)              # (1, G, 1, cD)
        pos_embed     = self.pos_embed[:, 1:, :].unsqueeze(1)        # (1, 1, L, pD)
        channel_embed = channel_embed.expand(-1, -1, L, -1)          # (1, G, L, cD)
        pos_embed     = pos_embed.expand(-1, G, -1, -1)              # (1, G, L, pD)
        pos_channel   = torch.cat((pos_embed, channel_embed), dim=-1) # (1, G, L, D)

        tokens = tokens + pos_channel    # (B, G, L, D)
        tokens = tokens.view(b, -1, D)   # (B, G*L, D)

        # Prepend CLS token
        cls_pos_channel = torch.cat(
            (self.pos_embed[:, :1, :], self.channel_cls_embed), dim=-1)
        cls_tokens = cls_pos_channel + self.cls_token.expand(b, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)  # (B, 1+G*L, D)
        tokens = self.pos_drop(tokens)

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        # Strip CLS -> (B, G*L, D)
        patch_tokens = tokens[:, 1:]

        # Average across G groups -> (B, L, D)
        patch_tokens = patch_tokens.view(b, G, L, D).mean(dim=1)

        # Reshape to spatial feature map
        H_out = W_out = int(L ** 0.5)   # 12 for 96x96, patch=8
        features = patch_tokens.reshape(b, H_out, W_out, D).permute(0, 3, 1, 2)
        return features   # (B, D, 12, 12)


# ============================================================================
# FPN Decoder -- identical to SpectralGPT model_cd_spectralgpt.py
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
# SatMAE CD Model
# ============================================================================

class SatMAE_CD(nn.Module):
    """
    Siamese SatMAE encoder + FPN change detection head.

    Forward pass:
        t1, t2 : (B, 6, 96, 96)
            -> shared GroupChannelsViTCD encoder
            -> f1, f2 : (B, 1024, 12, 12)
            -> diff = f1 - f2
            -> FPN conv pyramid + FPNHEAD -> (B, 256, 96, 96)
            -> cls_seg                    -> (B, 2,   96, 96) log-probs
    """

    def __init__(self):
        super().__init__()

        # Shared encoder with 6-band channel groups
        # ViT-Large to match pretrain-vit-large-e199.pth
        self.encoder = GroupChannelsViTCD(
            channel_embed=256,
            channel_groups=((0, 1, 2), (3, 4, 5)),
            img_size=96,
            patch_size=8,
            in_chans=6,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        # FPN conv pyramid: all branches from diff (B, 1024, 12, 12)
        self.conv0 = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.GroupNorm(32, 512), nn.GELU(),
            nn.ConvTranspose2d(512, 256, 8, 8),    # (B, 256, 96, 96)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.GroupNorm(32, 512), nn.GELU(),
            nn.ConvTranspose2d(512, 512, 4, 4),    # (B, 512, 48, 48)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, 1024, 1),
            nn.GroupNorm(32, 1024), nn.GELU(),
            nn.ConvTranspose2d(1024, 1024, 2, 2),  # (B, 1024, 24, 24)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1024, 2048, 1),
            nn.GroupNorm(32, 2048), nn.GELU(),     # (B, 2048, 12, 12)
        )

        # FPN decoder + binary classification head
        self.decoder = FPNHEAD(channels=2048, out_channels=256)
        self.cls_seg  = nn.Conv2d(256, 2, kernel_size=3, padding=1)
        self.sm       = nn.LogSoftmax(dim=1)

    def forward(self, t1, t2):
        """
        t1, t2  : (B, 6, 96, 96)
        returns : (B, 2, 96, 96) log-probabilities
        """
        f1   = self.encoder.forward_features_seg(t1)  # (B, 1024, 12, 12)
        f2   = self.encoder.forward_features_seg(t2)  # (B, 1024, 12, 12)
        diff = f1 - f2                                  # element-wise diff

        # multi-scale pyramid
        m = [
            self.conv0(diff),   # (B,  256, 96, 96)
            self.conv1(diff),   # (B,  512, 48, 48)
            self.conv2(diff),   # (B, 1024, 24, 24)
            self.conv3(diff),   # (B, 2048, 12, 12)
        ]

        x = self.decoder(m)     # (B, 256, 96, 96)
        x = self.cls_seg(x)     # (B, 2,   96, 96)
        return self.sm(x)


# ============================================================================
# Constructor + weight loading
# ============================================================================

def build_satmae_cd(pretrain_path: str = None) -> SatMAE_CD:
    """
    Build SatMAE CD model and optionally load pretrained encoder weights.

    Args:
        pretrain_path : path to SatMAE pretrained checkpoint
                        (e.g. checkpoint-199.pth from fMoW-Sentinel pretraining)
    Returns:
        SatMAE_CD instance
    """
    model = SatMAE_CD()

    if pretrain_path is not None:
        print(f"Load pre-trained checkpoint from: {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location='cpu')

        # SatMAE checkpoint is nested under 'model'
        if 'model' in checkpoint:
            ckpt = checkpoint['model']
        else:
            ckpt = checkpoint

        # Add 'encoder.' prefix to match our model structure
        # SatMAE checkpoint keys: patch_embed.0.proj.weight, blocks.0...
        # Our model keys:         encoder.patch_embed.0.proj.weight, encoder.blocks.0...
        ckpt_prefixed = {}
        for k, v in ckpt.items():
            # Skip decoder keys not needed for CD
            if any(k.startswith(p) for p in
                   ['decoder', 'mask_token', 'decoder_pred',
                    'decoder_embed', 'decoder_norm']):
                continue
            ckpt_prefixed[f'encoder.{k}'] = v

        # Adapt patch_embed weights via channel averaging.
        # Pretrained has 3 groups (e.g. 4+4+2 bands), CD model has 2 groups (3+3 bands).
        # Average pretrained channels -> expand to target channel count per group.
        state_dict = model.state_dict()
        channel_groups = ((0, 1, 2), (3, 4, 5))
        for i in range(len(channel_groups)):
            w_key = f'encoder.patch_embed.{i}.proj.weight'
            b_key = f'encoder.patch_embed.{i}.proj.bias'
            if w_key in ckpt_prefixed and w_key in state_dict:
                ckpt_w  = ckpt_prefixed[w_key]   # [D, C_pre, ph, pw]
                model_w = state_dict[w_key]        # [D, C_ft,  ph, pw]
                if ckpt_w.shape == model_w.shape:
                    print(f"  patch_embed.{i}: exact match, loading as-is")
                elif (ckpt_w.shape[0] == model_w.shape[0] and
                      ckpt_w.shape[2:] == model_w.shape[2:]):
                    C_ft  = model_w.shape[1]
                    avg_w = ckpt_w.mean(dim=1, keepdim=True)
                    ckpt_prefixed[w_key] = avg_w.expand(-1, C_ft, -1, -1).clone()
                    print(f"  patch_embed.{i}: adapted "
                          f"{ckpt_w.shape[1]} ch -> {C_ft} ch via channel averaging")
                else:
                    del ckpt_prefixed[w_key]
                    print(f"  patch_embed.{i}: incompatible shape, skipping")
            if b_key in ckpt_prefixed and b_key in state_dict:
                if ckpt_prefixed[b_key].shape != state_dict[b_key].shape:
                    del ckpt_prefixed[b_key]

        # Remove any remaining size-mismatched keys (e.g. channel_embed: 3 groups vs 2)
        state_dict = model.state_dict()
        mismatched = [k for k, v in ckpt_prefixed.items()
                      if k in state_dict and v.shape != state_dict[k].shape]
        for k in mismatched:
            print(f"  Skipping size mismatch: {k} "
                  f"(ckpt {ckpt_prefixed[k].shape} vs model {state_dict[k].shape})")
            del ckpt_prefixed[k]

        msg = model.load_state_dict(ckpt_prefixed, strict=False)
        print(msg)

    return model


if __name__ == '__main__':
    # Quick shape test
    # Run from ChangeDetection/ folder:
    # python src/model_cd_satmae.py
    model = build_satmae_cd(pretrain_path=None)
    t1  = torch.rand(2, 6, 96, 96)
    t2  = torch.rand(2, 6, 96, 96)
    out = model(t1, t2)
    print(f"Output shape: {out.shape}")    # expected: (2, 2, 96, 96)
    assert out.shape == (2, 2, 96, 96), "Shape mismatch!"
    print("Shape test passed.")
