
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(
        _os.path.join(_d, 'configs', 'paths.py')):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _os.path.join(_d, 'configs'))
from paths import DATA_ROOT, OUTPUT_ROOT, WEIGHTS, GFM_ROOT, PREDICTIONS  # noqa: E402

import os
import sys
import torch
import numpy as np

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
cudnn_benchmark = True
custom_imports = dict(imports=["geospatial_fm"])
num_frames = 3
img_size = 224
num_workers = 2
#num_workers = 0 #Debug


# class DebugShape:
#     def __init__(self, name):
#         self.name = name
    
#     def __call__(self, results):
#         img = results.get("img")
#         if img is not None:
#             if isinstance(img, torch.Tensor):
#                 print(f"[{self.name}] torch.Tensor shape: {img.shape}, dtype: {img.dtype}")
#             elif isinstance(img, np.ndarray):
#                 print(f"[{self.name}] numpy.ndarray shape: {img.shape}, dtype: {img.dtype}")
#             else:
#                 print(f"[{self.name}] type: {type(img)}")
#         return results

# model
# TO BE DEFINED BY USER: model path
pretrained_weights_path = os.path.expanduser(f"{WEIGHTS}/Prithvi_EO_V1_100M.pt")
num_layers = 6
patch_size = 16
embed_dim = 768
num_heads = 8
tubelet_size = 1
max_epochs = 80
eval_epoch_interval = 5

loss_weights_multi = [
    0.386375,
    0.661126,
    0.548184,
    0.640482,
    0.876862,
    0.925186,
    3.249462,
    1.542289,
    2.175141,
    2.272419,
    3.062762,
    3.626097,
    1.198702,
]
loss_func = dict(
    type="CrossEntropyLoss",
    use_sigmoid=False,
    class_weight=loss_weights_multi,
    avg_non_ignore=True,
)
output_embed_dim = embed_dim * num_frames


# TO BE DEFINED BY USER: Save directory
experiment = "prithvi_multi_temporal_crop_classification"
#project_dir = r"C:\MS_Research\experiments"
# work_dir = os.path.join(project_dir, experiment)

work_dir = os.path.expanduser(f"{OUTPUT_ROOT}/experiments/prithvi_multi_temporal_crop_classification/IA")
save_path = work_dir


dataset_type = "GeospatialDataset"

# TO BE DEFINED BY USER: data directory
data_root = os.path.expanduser(f"{DATA_ROOT}/multi_temporal_crop_segmentation")

# HLS stats
# img_norm_cfg = dict(
#     means=[
#         494.905781,
#         815.239594,
#         924.335066,
#         2968.881459,
#         2634.621962,
#         1739.579917,
#         494.905781,
#         815.239594,
#         924.335066,
#         2968.881459,
#         2634.621962,
#         1739.579917,
#         494.905781,
#         815.239594,
#         924.335066,
#         2968.881459,
#         2634.621962,
#         1739.579917,
#     ],
#     stds=[
#         284.925432,
#         357.84876,
#         575.566823,
#         896.601013,
#         951.900334,
#         921.407808,
#         284.925432,
#         357.84876,
#         575.566823,
#         896.601013,
#         951.900334,
#         921.407808,
#         284.925432,
#         357.84876,
#         575.566823,
#         896.601013,
#         951.900334,
#         921.407808,
#     ],
# )

# IA stats
img_norm_cfg = dict(
    means=[
        1861.19006065,  # Band 0
        2033.17032775,  # Band 1
        2273.37933660,  # Band 2
        3262.91588412,  # Band 3
        4457.44718789,  # Band 4
        3994.99188433,  # Band 5
        1704.63697798,  # Band 6
        1961.37926168,  # Band 7
        2034.19159947,  # Band 8
        3929.94252524,  # Band 9
        4352.22479367,  # Band 10
        3695.04113396,  # Band 11
        1518.83835315,  # Band 12
        1768.52085660,  # Band 13
        2016.60883454,  # Band 14
        3425.41515987,  # Band 15
        3619.43523780,  # Band 16
        2744.34036493,  # Band 17   
    ],
    stds=[
        307.48006869,  # Band 0
        351.67526808,  # Band 1
        447.86017086,  # Band 2
        654.81295100,  # Band 3
        697.75477528,  # Band 4
        788.08159599,  # Band 5
        341.76676414,  # Band 6
        357.44447699,  # Band 7
        513.43405597,  # Band 8
        842.65097823,  # Band 9
        920.07998270,  # Band 10
        1054.49693239,  # Band 11
        368.69514449,  # Band 12
        423.77628189,  # Band 13
        603.28624678,  # Band 14
        688.55751697,  # Band 15
        779.96415807,  # Band 16
        671.80796643,  # Band 17
    ],
)

# # CA stats
# img_norm_cfg = dict(
#     means=[
#         1399.18814785,  # Band 0
#         1624.09021280,  # Band 1
#         1530.61747529,  # Band 2
#         3433.65512704,  # Band 3
#         2625.27257993,  # Band 4
#         2036.76201361,  # Band 5
#         1690.31674406,  # Band 6
#         1965.86899305,  # Band 7
#         2231.22908437,  # Band 8
#         3764.86120507,  # Band 9
#         3585.95243342,  # Band 10
#         2720.77963287,  # Band 11
#         1547.22463694,  # Band 12
#         1832.03965290,  # Band 13
#         2079.88629347,  # Band 14
#         3402.57135275,  # Band 15
#         3377.24651308,  # Band 16
#         2538.78450119,  # Band 17
#     ],
#     stds=[
#         281.63779618,  # Band 0
#         300.33935069,  # Band 1
#         375.98413426,  # Band 2
#         1116.25457000,  # Band 3
#         684.00131194,  # Band 4
#         559.83401327,  # Band 5
#         382.69301356,  # Band 6
#         482.81761256,  # Band 7
#         787.29874717,  # Band 8
#         1098.04393375,  # Band 9
#         1231.90817451,  # Band 10
#         884.90944039,  # Band 11
#         349.85626067,  # Band 12
#         448.42248082,  # Band 13
#         723.75199174,  # Band 14
#         1019.93200015,  # Band 15
#         1201.66504561,  # Band 16
#         846.55948618,  # Band 17
#     ],
# )

# IL stats
# img_norm_cfg = dict(
#     means=[
#         1846.01041553,  # Band 0
#         2034.87291436,  # Band 1
#         2336.89914972,  # Band 2
#         3440.02626379,  # Band 3
#         4359.27030951,  # Band 4
#         3659.25810150,  # Band 5
#         1695.67070792,  # Band 6
#         2004.41327354,  # Band 7
#         2071.42573572,  # Band 8
#         4374.83196041,  # Band 9
#         4280.68264476,  # Band 10
#         3443.88865098,  # Band 11
#         1329.93065729,  # Band 12
#         1595.10951595,  # Band 13
#         1423.11484433,  # Band 14
#         4857.70613826,  # Band 15
#         2940.70615467,  # Band 16
#         1962.99561508,  # Band 17
#     ],
#     stds=[
#         283.66006748,  # Band 0
#         339.71900563,  # Band 1
#         474.30990502,  # Band 2
#         668.68752993,  # Band 3
#         840.73559942,  # Band 4
#         831.40692088,  # Band 5
#         346.60747720,  # Band 6
#         375.23409813,  # Band 7
#         594.97555072,  # Band 8
#         801.13507408,  # Band 9
#         1146.77483317,  # Band 10
#         1186.16430320,  # Band 11
#         216.80401250,  # Band 12
#         253.46961002,  # Band 13
#         308.23907609,  # Band 14
#         1013.56374670,  # Band 15
#         484.64192366,  # Band 16
#         388.80935123,  # Band 17
#     ],
# )

# bands = [0, 1, 2, 3, 4, 5]

# NC stats
# img_norm_cfg = dict(
#     means=[
#         1615.22474433,  # Band 0
#         1919.88049677,  # Band 1
#         1907.57302125,  # Band 2
#         4391.62877790,  # Band 3
#         3639.24284624,  # Band 4
#         2772.85214380,  # Band 5
#         1402.42608964,  # Band 6
#         1646.00419386,  # Band 7
#         1490.97300540,  # Band 8
#         4668.59382492,  # Band 9
#         3072.37737136,  # Band 10
#         2090.55866365,  # Band 11
#         1377.28325829,  # Band 12
#         1549.71437673,  # Band 13
#         1419.84248805,  # Band 14
#         4144.49420760,  # Band 15
#         2819.86398661,  # Band 16
#         1957.26925153,  # Band 17
#     ],
#     stds=[
#         445.83556981,  # Band 0
#         508.25045142,  # Band 1
#         765.69735543,  # Band 2
#         659.43243087,  # Band 3
#         1163.39733358,  # Band 4
#         1223.01516809,  # Band 5
#         276.18115909,  # Band 6
#         339.45512187,  # Band 7
#         419.55339594,  # Band 8
#         1044.42051202,  # Band 9
#         757.69613288,  # Band 10
#         650.64448343,  # Band 11
#         243.51106275,  # Band 12
#         291.01566979,  # Band 13
#         364.48130790,  # Band 14
#         920.12017449,  # Band 15
#         660.86197671,  # Band 16
#         555.53329805,  # Band 17
#     ],
# )

# MN stats
# img_norm_cfg = dict(
#     means=[
#         1683.21816572,  # Band 0
#         1795.12865640,  # Band 1
#         2027.99224864,  # Band 2
#         2614.43523828,  # Band 3
#         4018.17499175,  # Band 4
#         3795.35160064,  # Band 5
#         1332.62496374,  # Band 6
#         1593.52693158,  # Band 7
#         1510.57185140,  # Band 8
#         4645.33493779,  # Band 9
#         2915.19120058,  # Band 10
#         2091.37382972,  # Band 11
#         1444.89219603,  # Band 12
#         1713.57935808,  # Band 13
#         1806.45808454,  # Band 14
#         3524.61079859,  # Band 15
#         3447.32103589,  # Band 16
#         2790.36075006,  # Band 17
#     ],
#     stds=[
#         201.19653431,  # Band 0
#         253.75733532,  # Band 1
#         338.02958852,  # Band 2
#         492.03218955,  # Band 3
#         576.15203045,  # Band 4
#         595.78449640,  # Band 5
#         194.01547233,  # Band 6
#         226.79497442,  # Band 7
#         411.89867229,  # Band 8
#         1132.56897351,  # Band 9
#         477.13239916,  # Band 10
#         482.08074425,  # Band 11
#         203.44619838,  # Band 12
#         224.95563218,  # Band 13
#         366.10911266,  # Band 14
#         984.88041998,  # Band 15
#         610.66313299,  # Band 16
#         721.43276144,  # Band 17
#     ],
# )

bands = [0, 1, 2, 3, 4, 5]

tile_size = 224
orig_nsize = 512
crop_size = (tile_size, tile_size)
train_pipeline = [
    dict(type="LoadGeospatialImageFromFile", to_float32=True),
    dict(type="LoadGeospatialAnnotations", reduce_zero_label=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="ToTensor", keys=["img", "gt_semantic_seg"]),
    # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(type="TorchRandomCrop", crop_size=crop_size),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(bands), num_frames, tile_size, tile_size),
    ),
    dict(type="Reshape", keys=["gt_semantic_seg"], new_shape=(1, tile_size, tile_size)),
    dict(type="CastTensor", keys=["gt_semantic_seg"], new_type="torch.LongTensor"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]

test_pipeline = [
    dict(type="LoadGeospatialImageFromFile", to_float32=True),
    dict(type="ToTensor", keys=["img"]),
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(bands), num_frames, -1, -1),
        look_up=dict({"2": 1, "3": 2}),
    ),
    dict(type="CastTensor", keys=["img"], new_type="torch.FloatTensor"),
    dict(
        type="CollectTestList",
        keys=["img"],
        meta_keys=[
            "img_info",
            "seg_fields",
            "img_prefix",
            "seg_prefix",
            "filename",
            "ori_filename",
            "img",
            "img_shape",
            "ori_shape",
            "pad_shape",
            "scale_factor",
            "img_norm_cfg",
        ],
    ),
]

# test_pipeline = [
#     # Step 1: Load the 18-band image file. This also populates initial metadata.
#     dict(type='LoadGeospatialImageFromFile', to_float32=True),
#     # Step 2: Use MultiScaleFlipAug wrapper. This is the key.
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=None, # Use original chip size
#         img_ratios=[1.0], # Use original scale
#         flip=False, # No flipping
#         transforms=[
#             # These transforms apply ONLY to the image ('img')
#             # The mask data is not touched by this list.
#             dict(type='ToTensor', keys=['img']),
#             dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
#             dict(type='TorchNormalize', **img_norm_cfg),
#             dict(type="Reshape", 
#                  keys=["img"], 
#                  new_shape=(len(bands), num_frames, -1,-1),
#                  look_up={'2': 1, '3': 2}),
#             dict(type="CastTensor", keys=["img"], new_type="torch.FloatTensor"),

#             # This Collect step gathers the image and metadata *for the model*
#             # It replaces your old CollectTestList
#             dict(type='Collect', keys=['img'], # Collect img and mask
#                  meta_keys=("filename", "ori_filename", "img_shape", 
#                             "ori_shape", "pad_shape", "scale_factor", 
#                             "flip", "img_norm_cfg")), 
#         ])
# ]

CLASSES = (
    "Natural Vegetation",
    "Forest",
    "Corn",
    "Soybeans",
    "Wetlands",
    "Developed/Barren",
    "Open Water",
    "Winter Wheat",
    "Alfalfa",
    "Fallow/Idle Cropland",
    "Cotton",
    "Sorghum",
    "Other",
)

dataset = "GeospatialDataset"
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset,
        CLASSES=CLASSES,
        reduce_zero_label=True,
        data_root=data_root,
        img_dir="CentIA",
        ann_dir="CentIA",
        pipeline=train_pipeline,
        img_suffix="_merged.tif",
        seg_map_suffix=".mask.tif",
    ),
    val=dict(
        type=dataset,
        CLASSES=CLASSES,
        reduce_zero_label=True,
        data_root=data_root,
        img_dir="EastIA",
        ann_dir="EastIA",
        pipeline=test_pipeline,
        img_suffix="_merged.tif",
        seg_map_suffix=".mask.tif",
    ),
    test=dict(
        type=dataset,
        CLASSES=CLASSES,
        reduce_zero_label=True,
        data_root=data_root,
        img_dir="NWIA",
        ann_dir="NWIA",
        pipeline=test_pipeline,
        img_suffix="_merged.tif",
        seg_map_suffix=".mask.tif",
    ),
)

optimizer = dict(type="Adam", lr=1.5e-05, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)
log_config = dict(
    interval=10, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

checkpoint_config = dict(by_epoch=True, interval=100, out_dir=save_path)

evaluation = dict(
    interval=eval_epoch_interval,
    metric="mIoU",
    pre_eval=True,
    save_best="mIoU",
    by_epoch=True,
)
reduce_train_set = dict(reduce_train_set=False)
reduce_factor = dict(reduce_factor=1)
runner = dict(type="EpochBasedRunner", max_epochs=max_epochs)
workflow = [("train", 1)]
norm_cfg = dict(type="BN", requires_grad=True)

model = dict(
    type="TemporalEncoderDecoder",
    frozen_backbone=False,
    backbone=dict(
        type="TemporalViTEncoder",
        pretrained=pretrained_weights_path,
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=1,
        in_chans=len(bands),
        embed_dim=embed_dim,
        depth=6,
        num_heads=num_heads,
        mlp_ratio=4.0,
        norm_pix_loss=False,
    ),
    neck=dict(
        type="ConvTransformerTokensToEmbeddingNeck",
        embed_dim=embed_dim * num_frames,
        output_embed_dim=output_embed_dim,
        drop_cls_token=True,
        Hp=14,
        Wp=14,
    ),
    decode_head=dict(
        num_classes=len(CLASSES),
        in_channels=output_embed_dim,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=loss_func,
    ),
    auxiliary_head=dict(
        num_classes=len(CLASSES),
        in_channels=output_embed_dim,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=loss_func,
    ),
    train_cfg=dict(),
    test_cfg=dict(
        mode="slide",
        stride=(int(tile_size / 2), int(tile_size / 2)),
        crop_size=(tile_size, tile_size),
    ),
)
auto_resume = False