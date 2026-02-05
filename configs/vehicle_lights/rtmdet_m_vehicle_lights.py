"""
RTMDet-m Training Configuration for Vehicle Lights Detection
12 classes: front/rear lamp lenses (headlight, indicator, all-weather, brake, tailgate)

Based on:
- RTMDet-m_Training_Guide_Custom_Vehicle_Lights_v1.1_clean.md
- Claude_4_5_Playbook_Vehicle_Lights_Pipeline_Team_Guide_v1.3_clean.md

Key specifications:
- Model: RTMDet-m (medium variant)
- Input resolution: 1920×1080
- Classes: 12 vehicle light classes (exact order from playbook)
- Dataset: COCO format in data/vehicle_lights/
"""

# ======================== Base Configuration ========================
_base_ = [
    '../_base_/default_runtime.py'
]

# ======================== Model Settings ========================
# RTMDet-m architecture (medium variant)
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None
    ),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,  # RTMDet-m depth factor
        widen_factor=0.75,   # RTMDet-m width factor
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth'
        )
    ),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[192, 384, 768],  # RTMDet-m channels
        out_channels=192,             # RTMDet-m neck output
        num_csp_blocks=2,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)
    ),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=12,  # 12 vehicle light classes
        in_channels=192,
        feat_channels=192,
        stacked_convs=2,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        anchor_generator=dict(
            type='MlvlPointGenerator',
            offset=0,
            strides=[8, 16, 32]
        ),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=False
    ),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300
    )
)

# ======================== Dataset Settings ========================
# Canonical class list (EXACT order from playbook)
metainfo = dict(
    classes=(
        'front_headlight_left',
        'front_headlight_right',
        'front_indicator_left',
        'front_indicator_right',
        'front_all_weather_left',
        'front_all_weather_right',
        'rear_brake_left',
        'rear_brake_right',
        'rear_indicator_left',
        'rear_indicator_right',
        'rear_tailgate_left',
        'rear_tailgate_right'
    ),
    palette=[
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
        (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
        (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0)
    ]
)

dataset_type = 'CocoDataset'
data_root = 'data/vehicle_lights/'

# Backend args
backend_args = None

# ======================== Data Pipeline ========================
# Training pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    # CachedMosaic for data augmentation
    dict(
        type='CachedMosaic',
        img_scale=(1920, 1080),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False
    ),
    # RandomResize (conservative to avoid cutting lamps)
    dict(
        type='RandomResize',
        scale=(3840, 2160),  # 2x scale for mosaic
        ratio_range=(0.8, 1.2),  # Reduced range to preserve lamp visibility
        keep_ratio=True
    ),
    # RandomCrop (conservative)
    dict(
        type='RandomCrop',
        crop_size=(1920, 1080),
        allow_negative_crop=False  # Don't crop out all objects
    ),
    # Mild color jitter (bench environment)
    dict(type='YOLOXHSVRandomAug'),
    # RandomFlip
    dict(type='RandomFlip', prob=0.5),
    # Pad to fixed size
    dict(
        type='Pad',
        size=(1920, 1080),
        pad_val=dict(img=(114, 114, 114))
    ),
    dict(type='PackDetInputs')
]

# Training pipeline stage 2 (after epoch 280, no mosaic)
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(1920, 1080),
        ratio_range=(0.8, 1.2),
        keep_ratio=True
    ),
    dict(type='RandomCrop', crop_size=(1920, 1080)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Pad',
        size=(1920, 1080),
        pad_val=dict(img=(114, 114, 114))
    ),
    dict(type='PackDetInputs')
]

# Test/validation pipeline
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1920, 1080), keep_ratio=True),
    dict(
        type='Pad',
        size=(1920, 1080),
        pad_val=dict(img=(114, 114, 114))
    ),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# ======================== Dataloader Settings ========================
train_dataloader = dict(
    batch_size=8,  # Adjust based on GPU memory (A5000 should handle 8)
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)

test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)

# ======================== Evaluation Settings ========================
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
    proposal_nums=(100, 1, 10)
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
    proposal_nums=(100, 1, 10)
)

# ======================== Training Settings ========================
# Optimizer
base_lr = 0.004
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True
    )
)

# Learning rate scheduler
max_epochs = 300
stage2_num_epochs = 20

param_scheduler = [
    # Warmup
    dict(
        type='LinearLR',
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    # Cosine annealing
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True
    )
]

# Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)]
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ======================== Hooks ========================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=3,
        save_best='auto'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

custom_hooks = [
    # EMA hook
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49
    ),
    # Pipeline switch hook (disable mosaic in final epochs)
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2
    )
]

# ======================== Runtime Settings ========================
default_scope = 'mmdet'

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

# Load from pretrained COCO checkpoint (optional, for transfer learning)
load_from = None

# Resume training
resume = False

# Auto scale learning rate
auto_scale_lr = dict(base_batch_size=16, enable=False)
