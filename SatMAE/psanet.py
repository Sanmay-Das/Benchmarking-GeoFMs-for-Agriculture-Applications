import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange


# ============================================================================
# Pure PyTorch PSA mask -- replaces lib.psa.functional (C extension)
# No compilation needed
# ============================================================================

def psa_mask_fast(x, psa_type, mask_h, mask_w):
    """
    Vectorized pure PyTorch PSA mask generation.
    Replaces lib.psa.functional.psa_mask -- no C extension needed.

    Args:
        x:        (N, mask_h*mask_w, H, W) attention logits
        psa_type: 0 = collect, 1 = distribute
        mask_h:   attention mask height
        mask_w:   attention mask width

    Returns:
        (N, mask_h*mask_w, H, W) -- invalid positions zeroed out
    """
    N, C, H, W = x.shape
    assert C == mask_h * mask_w, \
        f"Channel dim {C} != mask_h*mask_w {mask_h*mask_w}"

    half_h = (mask_h - 1) // 2
    half_w = (mask_w - 1) // 2

    # Position indices
    h_idx  = torch.arange(H, device=x.device).view(H, 1, 1, 1)   # (H,1,1,1)
    w_idx  = torch.arange(W, device=x.device).view(1, W, 1, 1)   # (1,W,1,1)
    dh_idx = torch.arange(mask_h, device=x.device).view(1, 1, mask_h, 1)
    dw_idx = torch.arange(mask_w, device=x.device).view(1, 1, 1, mask_w)

    if psa_type == 0:
        # Collect: position (h,w) gathers from neighbour at offset (dh,dw)
        src_h = h_idx + dh_idx - half_h
        src_w = w_idx + dw_idx - half_w
        valid = ((src_h >= 0) & (src_h < H) &
                 (src_w >= 0) & (src_w < W))  # (H, W, mask_h, mask_w)
    else:
        # Distribute: position (h,w) distributes to neighbour at (dh,dw)
        dst_h = h_idx + dh_idx - half_h
        dst_w = w_idx + dw_idx - half_w
        valid = ((dst_h >= 0) & (dst_h < H) &
                 (dst_w >= 0) & (dst_w < W))  # (H, W, mask_h, mask_w)

    # valid: (H, W, mask_h, mask_w) -> (1, mask_h*mask_w, H, W)
    valid = valid.permute(2, 3, 0, 1).reshape(1, mask_h * mask_w, H, W).float()

    return x * valid


class PSA(nn.Module):
    def __init__(self, in_channels=2048, mid_channels=512, psa_type=2,
                 compact=False, shrink_factor=2, mask_h=11, mask_w=11,
                 normalization_factor=1.0, psa_softmax=True):
        super(PSA, self).__init__()
        assert psa_type in [0, 1, 2]
        self.psa_type            = psa_type
        self.compact             = compact
        self.shrink_factor       = shrink_factor
        self.mask_h              = mask_h
        self.mask_w              = mask_w
        self.psa_softmax         = psa_softmax
        self.normalization_factor = normalization_factor or (mask_h * mask_w)

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mask_h * mask_w, kernel_size=1, bias=False),
        )
        if psa_type == 2:
            self.reduce_p = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
            self.attention_p = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mask_h * mask_w, kernel_size=1, bias=False),
            )
        self.proj = nn.Sequential(
            nn.Conv2d(mid_channels * (2 if psa_type == 2 else 1),
                      in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x):
        out = x

        if self.psa_type in [0, 1]:
            x = self.reduce(x)
            n, c, h, w = x.size()
            if self.shrink_factor != 1:
                h = (h - 1) // self.shrink_factor + 1
                w = (w - 1) // self.shrink_factor + 1
                x = F.interpolate(x, size=(h, w), mode='bilinear',
                                  align_corners=True)
            y = self.attention(x)
            if self.compact:
                if self.psa_type == 1:
                    y = y.view(n, h*w, h*w).transpose(1, 2).view(n, h*w, h, w)
            else:
                y = psa_mask_fast(y, self.psa_type, self.mask_h, self.mask_w)
            if self.psa_softmax:
                y = F.softmax(y, dim=1)
            x = torch.bmm(
                x.view(n, c, h*w),
                y.view(n, h*w, h*w)
            ).view(n, c, h, w) * (1.0 / self.normalization_factor)

        elif self.psa_type == 2:
            x_col = self.reduce(x)
            x_dis = self.reduce_p(x)
            n, c, h, w = x_col.size()

            # Removed hardcoded assert h == 51 -- dynamic sizing

            if self.shrink_factor != 1:
                h = (h - 1) // self.shrink_factor + 1
                w = (w - 1) // self.shrink_factor + 1
                x_col = F.interpolate(x_col, size=(h, w), mode='bilinear',
                                      align_corners=True)
                x_dis = F.interpolate(x_dis, size=(h, w), mode='bilinear',
                                      align_corners=True)

            y_col = self.attention(x_col).float()
            y_dis = self.attention_p(x_dis).float()

            if self.compact:
                y_dis = y_dis.view(n, h*w, h*w).transpose(1, 2).view(n, h*w, h, w)
            else:
                y_col = psa_mask_fast(y_col, 0, self.mask_h, self.mask_w)
                y_dis = psa_mask_fast(y_dis, 1, self.mask_h, self.mask_w)

            if self.psa_softmax:
                y_col = F.softmax(y_col, dim=1)
                y_dis = F.softmax(y_dis, dim=1)

            x_col = torch.bmm(
                x_col.view(n, c, h*w),
                y_col.view(n, h*w, h*w)
            ).view(n, c, h, w) * (1.0 / self.normalization_factor)

            x_dis = torch.bmm(
                x_dis.view(n, c, h*w),
                y_dis.view(n, h*w, h*w)
            ).view(n, c, h, w) * (1.0 / self.normalization_factor)

            x = torch.cat([x_col, x_dis], 1)

        x = self.proj(x)
        if self.shrink_factor != 1:
            x = F.interpolate(x, size=(out.shape[2], out.shape[3]), 
                            mode='bilinear', align_corners=True)
        return torch.cat((out, x), 1)


class PSANet(nn.Module):
    def __init__(self, encoder, patch_size,
                 dropout=0.1, classes=13, zoom_factor=8,
                 use_psa=True, psa_type=2, compact=False,
                 shrink_factor=2, mask_h=11, mask_w=11,
                 normalization_factor=1.0, psa_softmax=True,
                 criterion=nn.CrossEntropyLoss(ignore_index=255)):
        """
        PSANet for crop segmentation with GroupChannels SatMAE encoder.

        Pure PyTorch -- no lib.psa.functional C extension needed.
        mask_h=11, mask_w=11 for 96x96 input:
            feature map = 96//8 = 12
            after shrink by 2: (12-1)//2+1 = 6
            mask = 2*6-1 = 11
        """
        super(PSANet, self).__init__()
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        assert psa_type in [0, 1, 2]

        self.zoom_factor = zoom_factor
        self.use_psa     = use_psa
        self.criterion   = criterion
        self.patch_size  = patch_size
        self.encoder     = encoder

        fea_dim = 1024  # ViT-Large embed_dim
        if use_psa:
            self.psa = PSA(fea_dim, 512, psa_type, compact, shrink_factor,
                           mask_h, mask_w, normalization_factor, psa_softmax)
            fea_dim *= 2  # PSA cat -> 2048

        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        self.aux = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(256, classes, kernel_size=1)
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))
        nwd = append_prefix_no_weight_decay("encoder.", self.encoder)
        if self.use_psa:
            nwd = nwd.union(
                append_prefix_no_weight_decay("psa.", self.psa)
            )
        return nwd

    def forward(self, x, is_train=True):
        """
        Args:
            x:        (B, 18, H, W) -- 18-band stacked multi-temporal input
            is_train: True  -> (main_logits, aux_logits) for MultiIoUBCE
                      False -> main_logits only for eval/inference
        """
        h, w = x.shape[2], x.shape[3]

        # GroupChannels encoder -> spatial feature map
        features  = self.encoder.forward_features_seg(x)
        x_spatial = features[0]  # (B, 1024, 12, 12)
        x_tmp     = x_spatial    # save for aux head

        if self.use_psa:
            x_spatial = self.psa(x_spatial)  # (B, 2048, 12, 12)

        x_out = self.cls(x_spatial)
        x_out = F.interpolate(x_out, size=(h, w), mode='bilinear',
                              align_corners=True)

        if is_train:
            aux = self.aux(x_tmp)
            aux = F.interpolate(aux, size=(h, w), mode='bilinear',
                                align_corners=True)
            return x_out, aux
        else:
            return x_out