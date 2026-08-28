"""
models_satmae_fcn.py
--------------------
SatMAE ViT-Large encoder + Prithvi-style FCNHead decoder for crop segmentation.

Encoder : GroupChannels ViT-Large (models_vit_group_channels.py)
          Input  : (B, 18, 96, 96)  --  6 bands x 3 timesteps
          Output : (B, 1024, 12, 12) via forward_features_seg()

Decoder : FCNHead -- adapted from Prithvi EO V1 (mmseg FCNHead)
          Prithvi config: in_channels=2304 (768*3), channels=256
          Adapted  here: in_channels=1024 (SatMAE ViT-Large embed_dim)

          Main head : num_convs=1, Conv2d(1024->256,3x3)+BN+ReLU
                      -> Dropout2d(0.1) -> Conv2d(256->14,1x1)
                      -> bilinear upsample -> (B, 14, 96, 96)

          Aux head  : num_convs=2, Conv2d(1024->256,3x3)+BN+ReLU
                      -> Conv2d(256->256,3x3)+BN+ReLU
                      -> Dropout2d(0.1) -> Conv2d(256->14,1x1)
                      -> bilinear upsample -> (B, 14, 96, 96)

Reference : Prithvi EO V1 multi_temporal_crop_classification.py
            mmseg/models/decode_heads/fcn_head.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNHead(nn.Module):
    """
    Standalone FCNHead -- matches Prithvi's mmseg FCNHead exactly.

    Args:
        in_channels  : input feature channels (1024 for SatMAE ViT-Large)
        channels     : intermediate channels (256 -- Prithvi config)
        num_convs    : number of 3x3 conv layers (1=main, 2=aux -- Prithvi config)
        dropout_ratio: Dropout2d before cls_seg (0.1 -- Prithvi config)
        nb_classes   : number of output classes (14: 0=NoData, 1-13=crops)
        align_corners: bilinear upsample flag (False -- Prithvi config)
    """

    def __init__(self, in_channels=1024, channels=256, num_convs=1,
                 dropout_ratio=0.1, nb_classes=14, align_corners=False):
        super().__init__()
        self.align_corners = align_corners

        # Stack of conv+BN+ReLU blocks
        convs = []
        for i in range(num_convs):
            _in = in_channels if i == 0 else channels
            convs.append(nn.Sequential(
                nn.Conv2d(_in, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ))
        self.convs = nn.Sequential(*convs)

        # Dropout2d before classification (Prithvi: dropout_ratio=0.1)
        self.dropout = nn.Dropout2d(dropout_ratio)

        # Classification conv
        self.cls_seg = nn.Conv2d(channels, nb_classes, kernel_size=1)

    def forward(self, x):
        """x: (B, in_channels, H, W)  ->  (B, nb_classes, H, W)"""
        x = self.convs(x)
        x = self.dropout(x)
        return self.cls_seg(x)


class SatMAEFCN(nn.Module):
    """
    SatMAE ViT-Large + Prithvi FCNHead decoder.

    Follows Prithvi EO V1 segmentation setup:
      - Main head : num_convs=1
      - Aux  head : num_convs=2
      - Both use channels=256, dropout=0.1, BN, align_corners=False
    """

    def __init__(self, encoder, nb_classes=14):
        super().__init__()
        self.encoder = encoder
        embed_dim = 1024  # ViT-Large embed dim

        # Main head -- num_convs=1 (Prithvi config)
        self.main_head = FCNHead(
            in_channels=embed_dim,
            channels=256,
            num_convs=1,
            dropout_ratio=0.1,
            nb_classes=nb_classes,
            align_corners=False,
        )

        # Auxiliary head -- num_convs=2 (Prithvi config)
        self.aux_head = FCNHead(
            in_channels=embed_dim,
            channels=256,
            num_convs=2,
            dropout_ratio=0.1,
            nb_classes=nb_classes,
            align_corners=False,
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, is_train=True):
        H, W = x.shape[2], x.shape[3]

        # GroupChannels encoder -> (B, 1024, 12, 12)
        feat = self.encoder.forward_features_seg(x)[0]

        # Main head
        main_logits = self.main_head(feat)                        # (B, 14, 12, 12)
        main_logits = F.interpolate(
            main_logits, size=(H, W),
            mode='bilinear', align_corners=False)                  # (B, 14, 96, 96)

        if is_train:
            # Aux head -- same feature map, different conv stack
            aux_logits = self.aux_head(feat)                      # (B, 14, 12, 12)
            aux_logits = F.interpolate(
                aux_logits, size=(H, W),
                mode='bilinear', align_corners=False)              # (B, 14, 96, 96)
            return main_logits, aux_logits

        return main_logits
