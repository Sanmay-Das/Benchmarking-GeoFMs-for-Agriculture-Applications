"""
model_cd_spectralgpt.py
-----------------------
SpectralGPT Change Detection model adapted for your benchmark.

Architecture (mirrors SpectralGPT paper Fig. 5 + Section 3.4):
    Shared ViT encoder (SpectralGPT pretrained)
        T1 -> encoder -> f1
        T2 -> encoder -> f2  (same shared weights)
        diff = f1 - f2
        diff -> reshape -> conv pyramid -> FPNHEAD -> cls_seg -> LogSoftmax

Key adaptation from their OSCD model:
    Their OSCD : 12 bands, num_frames=12, t_patch_size=3
    Yours      :  6 bands, num_frames=6,  t_patch_size=3
                  6/3 = 2 temporal tokens -> fc(2->1)

Place this file at:
    downstream_tasks/ChangeDetection/src/model_cd_spectralgpt.py

Imports used in train_cd_spectralgpt.py:
    from src.model_cd_spectralgpt import build_spectralgpt_cd
"""

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

# These live in downstream_tasks/ChangeDetection/util/
# (copied from SegMunich/util/)
from util.video_vit import Attention, Block, PatchEmbed
from util.pos_embed import interpolate_pos_embed


# ============================================================================
# FPN Decoder (identical to SpectralGPT OSCD models_vit_tensor_CD.py)
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
# SpectralGPT CD Model
# ============================================================================

class SpectralGPT_CD(nn.Module):
    """
    Siamese SpectralGPT encoder + FPN change detection head.

    num_frames=6, t_patch_size=3 -> T = 6/3 = 2 temporal tokens per location
    img_size=128, patch_size=8   -> 16x16 = 256 spatial tokens
    embed_dim=768                -> ViT-Base
    """

    def __init__(
        self,
        img_size=128,
        patch_size=8,
        in_chans=1,
        num_frames=6,
        t_patch_size=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        no_qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        sep_pos_embed=True,
        **kwargs,
    ):
        super().__init__()

        self.sep_pos_embed = sep_pos_embed

        # patch embedding (shared between T1 and T2)
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim, num_frames, t_patch_size
        )
        input_size      = self.patch_embed.input_size   # (T=2, H_tok=16, W_tok=16)
        self.input_size = input_size
        num_patches     = self.patch_embed.num_patches

        # positional embeddings
        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], embed_dim))
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim))
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim), requires_grad=True)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, 0.5, depth)]
        self.blocks = nn.ModuleList([
            Block(
                embed_dim, num_heads, mlp_ratio,
                qkv_bias=not no_qkv_bias,
                qk_scale=None,
                norm_layer=norm_layer,
                drop_path=dpr[i],
                attn_func=partial(Attention, input_size=input_size),
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # temporal token collapse: T tokens -> 1 per spatial location
        # T = num_frames / t_patch_size = 6 / 3 = 2
        T = input_size[0]
        self.fc = nn.Linear(T, 1)

        # spatial token grid size
        self.H_tok = input_size[1]   # 16
        self.W_tok = input_size[2]   # 16

        # conv pyramid over difference features
        self.conv0 = nn.Sequential(
            nn.Conv2d(embed_dim, 512, 1),
            nn.GroupNorm(32, 512), nn.GELU(),
            nn.ConvTranspose2d(512, 256, 8, 8),    # -> (B, 256, 128, 128)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, 512, 1),
            nn.GroupNorm(32, 512), nn.GELU(),
            nn.ConvTranspose2d(512, 512, 4, 4),    # -> (B, 512,  64,  64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(embed_dim, 1024, 1),
            nn.GroupNorm(32, 1024), nn.GELU(),
            nn.ConvTranspose2d(1024, 1024, 2, 2),  # -> (B,1024,  32,  32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(embed_dim, 2048, 1),
            nn.GroupNorm(32, 2048), nn.GELU(),     # -> (B,2048,  16,  16)
        )

        # FPN decoder + binary classification head
        self.decoder = FPNHEAD()
        self.cls_seg  = nn.Conv2d(256, 2, kernel_size=3, padding=1)
        self.sm       = nn.LogSoftmax(dim=1)

    # ------------------------------------------------------------------
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'pos_embed_spatial', 'pos_embed_temporal'}

    # ------------------------------------------------------------------
    def _encode(self, x):
        """
        Encode a single-date 6-band image through the shared ViT encoder.
        x : (B, 6, H, W)
        returns tokens : (B, T, L, C)
        """
        B = x.shape[0]

        # SpectralGPT PatchEmbed expects (B, in_chans, num_frames, H, W)
        # in_chans=1, num_frames=6 -> unsqueeze channel dim
        x = x.unsqueeze(1)           # (B, 1, 6, H, W)
        x = self.patch_embed(x)      # (B, T, L, C)
        N, T, L, C = x.shape

        x = x.view(B, T * L, C)     # (B, T*L, C)

        # add positional embedding
        if self.sep_pos_embed:
            pos = (self.pos_embed_spatial.repeat(1, self.input_size[0], 1)
                   + torch.repeat_interleave(
                       self.pos_embed_temporal,
                       self.input_size[1] * self.input_size[2], dim=1))
        else:
            pos = self.pos_embed
        x = x + pos

        # reshape to (B, T, L, C) if blocks require it
        requires_t = (len(self.blocks) > 0
                      and hasattr(self.blocks[0].attn, 'requires_t_shape')
                      and self.blocks[0].attn.requires_t_shape)
        if requires_t:
            x = x.view(B, T, L, C)

        for blk in self.blocks:
            x = blk(x)

        x = x.view(B, T, L, C)
        return x    # (B, T=2, L=256, C=768)

    # ------------------------------------------------------------------
    def forward(self, t1, t2):
        """
        t1, t2  : (B, 6, H, W)
        returns : (B, 2, H, W) log-probabilities
        """
        B = t1.shape[0]

        # encode both dates with shared encoder
        f1 = self._encode(t1)    # (B, T, L, C)
        f2 = self._encode(t2)    # (B, T, L, C)

        # collapse temporal tokens: (B, L, C, T) -> fc -> (B, L, C)
        f1 = f1.permute(0, 2, 3, 1)   # (B, L, C, T)
        f2 = f2.permute(0, 2, 3, 1)

        if f1.shape[3] > 1:
            f1 = self.fc(f1).squeeze(-1)   # (B, L, C)
            f2 = self.fc(f2).squeeze(-1)
        else:
            f1 = f1.squeeze(-1)
            f2 = f2.squeeze(-1)

        # element-wise difference (Siamese subtraction)
        diff = f1 - f2    # (B, L=256, C=768)

        # reshape to spatial feature map
        diff = diff.reshape(B, self.H_tok, self.W_tok, -1)  # (B,16,16,768)
        diff = diff.permute(0, 3, 1, 2).contiguous()         # (B,768,16,16)

        # multi-scale pyramid
        m = [
            self.conv0(diff),   # (B, 256, 128, 128)
            self.conv1(diff),   # (B, 512,  64,  64)
            self.conv2(diff),   # (B,1024,  32,  32)
            self.conv3(diff),   # (B,2048,  16,  16)
        ]

        # FPN decode + classify
        x = self.decoder(m)     # (B, 256, 128, 128)
        x = self.cls_seg(x)     # (B, 2,   128, 128)
        x = self.sm(x)          # log-softmax
        return x


# ============================================================================
# Constructor + weight loading
# ============================================================================

def build_spectralgpt_cd(pretrain_path: str = None) -> SpectralGPT_CD:
    """
    Build SpectralGPT CD model and optionally load pretrained encoder weights.

    Args:
        pretrain_path : path to SpectralGPT+.pth
    Returns:
        SpectralGPT_CD instance
    """
    model = SpectralGPT_CD(
        img_size=128,
        patch_size=8,
        in_chans=1,
        num_frames=6,
        t_patch_size=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        sep_pos_embed=True,
    )

    if pretrain_path is not None:
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        print(f"Load pre-trained checkpoint from: {pretrain_path}")

        ckpt_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state_dict = model.state_dict()

        # Remove keys that will mismatch
        # (same logic as your segmentation train_custom_spectralgpt.py)
        keys_to_check = [
            'pos_embed',
            'pos_embed_spatial',
            'pos_embed_temporal',       # pretrained T=4, ours T=2
            'patch_embed.proj.weight',
            'patch_embed.proj.bias',
            'head.weight',
            'head.bias',
            'fc.0.weight',              # pretrained T=4, ours T=2
            'fc.0.bias',
        ]
        for k in keys_to_check:
            if k in ckpt_model and k in state_dict:
                if ckpt_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    print(f"  Pretrained shape: {ckpt_model[k].shape}")
                    print(f"  Current shape:    {state_dict[k].shape}")
                    del ckpt_model[k]

        interpolate_pos_embed(model, ckpt_model)

        msg = model.load_state_dict(ckpt_model, strict=False)
        print(msg)

    return model


if __name__ == '__main__':
    # Quick shape test — run from ChangeDetection/ folder:
    # python src/model_cd_spectralgpt.py
    model = build_spectralgpt_cd(pretrain_path=None)
    t1  = torch.rand(2, 6, 128, 128)
    t2  = torch.rand(2, 6, 128, 128)
    out = model(t1, t2)
    print(f"Output shape: {out.shape}")   # expected: (2, 2, 128, 128)
    assert out.shape == (2, 2, 128, 128), "Shape mismatch!"
    print("Shape test passed.")