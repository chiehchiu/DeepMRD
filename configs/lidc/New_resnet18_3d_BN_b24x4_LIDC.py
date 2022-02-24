#_base_ = [
    #'../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py',
    #'../_base_/schedules/imagenet_bs1024_linearlr_bn_nowd.py',
    #'../_base_/default_runtime.py'
#]
fp16 = dict(loss_scale=512.)
# model settings
model = dict(
    type='ImageClassifier',
    #pretrained='pretrained_models/res3d18_imagenet_BN-3e251889.pth',
    pretrained='pretrained_models/res3d18_imagenet_BN-73.43-final.pth',
    backbone=dict(
        type='ResNet3D',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        depth_stride=True,      # consistent with ACS
        stem_stride=True,
        strides=(1, 2, 2, 2),   # consistent with ACS
        in_channels=1,
        conv_cfg=dict(type='Conv3d'),
        #norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        norm_cfg=dict(type='BN3d'),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling', use_3d_gap=True),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, ),
    ))
# dataset settings
dataset_type = 'LIDCNewDataset'
img_norm_cfg = dict(
    #mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    mean=[114.495]*3, std=[57.63]*3, to_rgb=True)

train_pipeline = [
    # 1. random crop 2. rotation 3. reflection(flip by 3 axis)
    dict(type='LoadTensorFromPNG'),
    dict(type='TensorNormCropRotateFlip', crop_size=48, move=5, train=True),
    #dict(type='RandomResizedCrop', size=224),
    #dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    #dict(type='Normalize', **img_norm_cfg),
    dict(type='ToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadTensorFromPNG'),
    dict(type='TensorNormCropRotateFlip', crop_size=48, move=5, train=False),
    #dict(type='Resize', size=(256, -1)),
    #dict(type='CenterCrop', crop_size=224),
    #dict(type='Normalize', **img_norm_cfg),
    dict(type='ToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=24,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/LIDC/',
        ann_file='train_pair_2665_0.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/LIDC/',
        ann_file='validate_pair_2665_0.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/LIDC/nodule',
        ann_file='test_pair_2665_0.txt',
        pipeline=test_pipeline))

evaluation = dict(interval=2, metric='all',
        metric_options=dict(topk=(1, 2)) )

# optimizer
#optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
#optimizer = dict(type='Adam', lr=0.0005, weight_decay=0.0001)
#optimizer_config = dict(grad_clip=None)
#lr_config = dict(policy='step', step=[50, 75])
#runner = dict(type='EpochBasedRunner', max_epochs=100)


# checkpoint saving
checkpoint_config = dict(interval=50)
# yapf:disable
log_config = dict(
    interval=7,
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

#optimizer = dict(
    #type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001, paramwise_cfg=dict(norm_decay_mult=0))
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)
#optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy='step', step=[30, 60, 90], warmup='constant', warmup_iters=50)
#lr_config = dict(policy='step', step=[50, 75], warmup='constant', warmup_iters=50)
#lr_config = dict(policy='step', step=[30, 40], warmup='constant', warmup_iters=50,)
#lr_config = dict(
    #policy='CosineAnnealing', min_lr=1e-6, warmup='linear', warmup_iters=50, warmup_ratio=0.25)
#lr_config = dict( policy='poly', min_lr=1e-4, by_epoch=False,
    #warmup='constant', warmup_iters=100, )
runner = dict(type='EpochBasedRunner', max_epochs=100)
