#_base_ = [
    #'../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py',
    #'../_base_/schedules/imagenet_bs256.py',
    #'../_base_/default_runtime.py'
#]
fp16 = dict(loss_scale=512.)
# model settings
model = dict(
    type='ImageClassifier',
    #pretrained='torchvision://resnet18',
    pretrained='pretrained_model/resnet18-5c106cde_med1channel.pth',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        #depth_stride=True,      # consistent with ACS
        stem_stride=False,       # consistent with ACS
        strides=(1, 2, 1, 2),   # consistent with ACS
        in_channels=1,
        norm_cfg=dict(type='BN'),
        style='pytorch'),
    converter='2.5D', # convert 2d backbone to 3d ones with I3D method.
    neck=dict(type='GlobalAveragePooling', use_3d_gap=True),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 2),
    ))
# dataset settings
dataset_type = 'LIDCDataset'
img_norm_cfg = dict(
    #mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    mean=[114.495]*3, std=[57.63]*3, to_rgb=True)

train_pipeline = [
    # 1. random crop 2. rotation 3. reflection(flip by 3 axis)
    dict(type='LoadTensorFromFile'),
    dict(type='TensorNormCropRotateFlip', crop_size=48, move=5, train=True),
    #dict(type='RandomResizedCrop', size=224),
    #dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    #dict(type='Normalize', **img_norm_cfg),
    dict(type='ToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadTensorFromFile'),
    dict(type='TensorNormCropRotateFlip', crop_size=48, move=5, train=False),
    #dict(type='Resize', size=(256, -1)),
    #dict(type='CenterCrop', crop_size=224),
    #dict(type='Normalize', **img_norm_cfg),
    dict(type='ToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/lidc/',
        ann_file='train_test_split.csv',
        sub_set='info/lidc_nodule_info_new_with_subset.csv',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/lidc/',
        ann_file='train_test_split.csv',
        sub_set='info/lidc_nodule_info_new_with_subset.csv',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/lidc/nodule',
        ann_file='train_test_split.csv',
        sub_set='info/lidc_nodule_info_new_with_subset.csv',
        pipeline=test_pipeline))

evaluation = dict(interval=5, metric='all',
        metric_options=dict(topk=(1, 2)) )

# optimizer
#optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
#lr_config = dict(policy='step', step=[30, 60, 90])
lr_config = dict(policy='step', step=[50, 75])
runner = dict(type='EpochBasedRunner', max_epochs=100)

# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
