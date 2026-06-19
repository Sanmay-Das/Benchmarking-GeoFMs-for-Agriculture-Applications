"""
models_satmae_fpn.py
--------------------
SatMAE ViT-Large encoder + SpectralGPT-style FPN decoder for crop segmentation.

Encoder : GroupChannels ViT-Large (models_vit_group_channels.py)
          Input  : (B, 18, 96, 96)  —  6 bands × 3 timesteps
          Output : (B, 1024, 12, 12) via forward_features_seg()

Decoder : 4-scale FPN (same design as SpectralGPT SegMunich src/)
          conv0 : 1024 →  256 @ 8× upsample → (B,  256, 96, 96)
          conv1 : 1024 →  512 @ 4× upsample → (B,  512, 48, 48)
          conv2 : 1024 → 1024 @ 2× upsample → (B, 1024, 24, 24)
          conv3 : 1024 → 2048 @ 1× (none)   → (B, 2048, 12, 12)
          FPNHEAD fuses all four levels → cls_seg → (B, nb_classes, 96, 96)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── PPM / PPMHEAD / FPNHEAD ── copied verbatim from SpectralGPT SegMunich ────

class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super().__init__()
        for pool_size in pool_sizes:
            self.append(nn.Sequential(
                nn.AdaptiveMaxPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            ))

    def forward(self, x):
        outs = []
        for ppm in self:
            outs.append(F.interpolate(ppm(x), size=(x.size(2), x.size(3)),
                                      mode='bilinear', align_corners=True))
        return outs


class PPMHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.psp_modules = PPM(pool_sizes, in_channels, out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * out_channels,
                      out_channels, kernel_size=1),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        return self.final(torch.cat(out, dim=1))


class FPNHEAD(nn.Module):
    """
    4-level FPN head — identical to SpectralGPT SegMunich FPNHEAD.
    Expects input_fpn = [fpn0, fpn1, fpn2, fpn3] with channels
    [channels//8, channels//4, channels//2, channels] = [256, 512, 1024, 2048].
    """
    def __init__(self, channels=2048, out_channels=256):
        super().__init__()
        self.PPMHead = PPMHEAD(channels, out_channels)

        self.Conv_fuse1  = nn.Sequential(
            nn.Conv2d(channels // 2, out_channels, 1),
            nn.GroupNorm(16, out_channels), nn.GELU(), nn.Dropout(0.5))
        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GroupNorm(16, out_channels), nn.GELU(), nn.Dropout(0.5))

        self.Conv_fuse2  = nn.Sequential(
            nn.Conv2d(channels // 4, out_channels, 1),
            nn.GroupNorm(16, out_channels), nn.GELU(), nn.Dropout(0.5))
        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GroupNorm(16, out_channels), nn.GELU(), nn.Dropout(0.5))

        self.Conv_fuse3  = nn.Sequential(
            nn.Conv2d(channels // 8, out_channels, 1),
            nn.GroupNorm(16, out_channels), nn.GELU(), nn.Dropout(0.5))
        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GroupNorm(16, out_channels), nn.GELU(), nn.Dropout(0.5))

        self.fuse_all = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            nn.GroupNorm(16, out_channels), nn.GELU(), nn.Dropout(0.5))

        self.conv_x1 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, input_fpn):
        # input_fpn[-1] = (B, 2048, 12, 12)  — coarsest, richest semantics
        # input_fpn[-4] = (B,  256, 96, 96)  — finest, highest resolution
        x1 = self.PPMHead(input_fpn[-1])                                   # 256 @ 12×12

        x  = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x  = self.conv_x1(x) + self.Conv_fuse1(input_fpn[-2])             # 256 @ 24×24
        x2 = self.Conv_fuse1_(x)

        x  = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        x  = x + self.Conv_fuse2(input_fpn[-3])                           # 256 @ 48×48
        x3 = self.Conv_fuse2_(x)

        x  = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x  = x + self.Conv_fuse3(input_fpn[-4])                           # 256 @ 96×96
        x4 = self.Conv_fuse3_(x)

        # Align all levels to finest resolution and fuse
        x1 = F.interpolate(x1, size=x4.shape[-2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=x4.shape[-2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x4.shape[-2:], mode='bilinear', align_corners=True)
        return self.fuse_all(torch.cat([x1, x2, x3, x4], dim=1))          # 256 @ 96×96


# ── SatMAEFPN ─────────────────────────────────────────────────────────────────

class SatMAEFPN(nn.Module):
    """
    SatMAE ViT-Large + SpectralGPT FPN decoder.
    Drop-in replacement for PSANet in the SatMAE segmentation pipeline.
    """

    def __init__(self, encoder, nb_classes=14):
        super().__init__()
        self.encoder  = encoder
        embed_dim = 1024  # ViT-Large embed dim

        # Multi-scale projection heads — SpectralGPT design, adapted 768→1024 input
        self.conv0 = nn.Sequential(
            nn.Conv2d(embed_dim, 512, kernel_size=1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 256, kernel_size=8, stride=8),   # 12→96
            nn.Dropout(0.5),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, 512, kernel_size=1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=4),   # 12→48
            nn.Dropout(0.5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(embed_dim, 1024, kernel_size=1),
            nn.GroupNorm(32, 1024),
            nn.GELU(),
            nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2), # 12→24
            nn.Dropout(0.5),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(embed_dim, 2048, kernel_size=1),
            nn.GroupNorm(32, 2048),
            nn.GELU(),
            nn.Dropout(0.5),                                          # 12→12 (no upsample)
        )

        self.decoder = FPNHEAD(channels=2048, out_channels=256)
        self.cls_seg  = nn.Conv2d(256, nb_classes, kernel_size=3, padding=1)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        # GroupChannels encoder → (B, 1024, 12, 12)
        feat = self.encoder.forward_features_seg(x)[0]

        # 4-scale projections from the same feature map
        m0 = self.conv0(feat)   # (B,  256, 96, 96)
        m1 = self.conv1(feat)   # (B,  512, 48, 48)
        m2 = self.conv2(feat)   # (B, 1024, 24, 24)
        m3 = self.conv3(feat)   # (B, 2048, 12, 12)

        # FPN fusion
        fused  = self.decoder([m0, m1, m2, m3])        # (B, 256, 96, 96)
        logits = self.cls_seg(fused)                    # (B, nb_classes, 96, 96)

        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W),
                                   mode='bilinear', align_corners=True)
        return logits
