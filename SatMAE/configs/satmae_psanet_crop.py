"""
MMSegmentation Config: SatMAE + PSANet for Crop Segmentation
Using MMSeg 0.30.0 + MMCV 1.7.1
"""

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(
        _os.path.join(_d, 'configs', 'paths.py')):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _os.path.join(_d, 'configs'))
from paths import DATA_ROOT, WEIGHTS, MSR_ROOT, OUTPUT_ROOT, PREDICTIONS  # noqa: E402

custom_imports = dict(
    imports=['satmae_backbone', 'geospatial_fm', 'custom_pipelines'],
    allow_failed_imports=False
)

# class DebugShapes:
#     """Debug transform to print shapes"""
#     def __call__(self, results):
#         print(f"[DEBUG PIPELINE] img shape: {results['img'].shape if 'img' in results else 'N/A'}")
#         print(f"[DEBUG PIPELINE] gt_semantic_seg shape: {results['gt_semantic_seg'].shape if 'gt_semantic_seg' in results else 'N/A'}")
#         return results

# from mmseg.datasets.builder import PIPELINES
# PIPELINES.register_module()(DebugShapes)

# Dataset settings
dataset_type = 'CustomDataset'
data_root = f'{DATA_ROOT}/SatMAE_chips_multitemporal/CentIA'
img_norm_cfg = dict(means=[0]*18, stds=[1]*18)
crop_size = (96, 96)
num_classes = 13

classes = [
    'Natural Vegetation',
    'Forest',
    'Corn',
    'Soybeans',
    'Wetlands',
    'Developed/Barren',
    'Open Water',
    'Winter Wheat',
    'Alfalfa',
    'Fallow/Idle Cropland',
    'Cotton',
    'Sorghum',
    'Other'
]


# Model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    
    # SatMAE ViT-Large Backbone
    backbone=dict(
        type='SatMAEBackbone',
        pretrained=f'{WEIGHTS}/pretrain-vit-large-e199.pth',
        img_size=96,
        patch_size=8,
        in_chans=18,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        channel_groups=[[0, 1, 2, 6], [3, 4, 5, 7], [8, 9]],
        drop_path_rate=0.2,
    ),
    
    # PSANet Decode Head
    decode_head=dict(
        type='PSAHead',
        in_channels=1024,
        in_index=0,
        channels=512,
        psa_type='bi-direction',
        compact=False,
        shrink_factor=2,
        mask_size=(12, 12),
        normalization_factor=1.0,
        psa_softmax=True,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),
    
    # Auxiliary head
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=0,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4
        )
    ),
    
    # Training and testing
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# Pipeline for MMSeg 0.30.0
train_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=True),
    dict(type='LoadGeospatialAnnotations', reduce_zero_label=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='ToTensor', keys=['img', 'gt_semantic_seg']),
    dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
    dict(type='TorchNormalize', **img_norm_cfg),
    dict(type='TorchRandomCrop', crop_size=(96, 96)),
    dict(type='Reshape', new_shape=(-1, 96, 96), keys=['gt_semantic_seg'], look_up=None),
    dict(type='CastTensor', keys=['gt_semantic_seg'], new_type='torch.LongTensor'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=True),
    dict(type='ToTensor', keys=['img']),
    dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
    dict(type='TorchNormalize', **img_norm_cfg),
    dict(
        type='CollectTestList', 
        keys=['img'],
        meta_keys=[
            'filename', 'ori_filename', 'ori_shape', 'img_shape', 
            'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg'
        ]
    )
]

# Dataloaders for MMSeg 0.30.0
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        ann_dir='',
        img_suffix='.tif',
        seg_map_suffix='_mask.tif',
        split='train.txt',
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        ann_dir='',
        img_suffix='.tif',
        seg_map_suffix='_mask.tif',
        split='val.txt',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        ann_dir='',
        img_suffix='.tif',
        seg_map_suffix='_mask.tif',
        split='test.txt',
        classes=classes,
        pipeline=test_pipeline
    )
)

# Optimizer (SGD following paper)
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1)}
    )
)

optimizer_config = dict()

# Learning rate policy
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=0,
    by_epoch=True
)

# Runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=30)
checkpoint_config = dict(by_epoch=True, interval=5)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True)

# Logging
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True)
    ]
)

# Misc
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
work_dir = './work_dirs/satmae_psanet_crop'