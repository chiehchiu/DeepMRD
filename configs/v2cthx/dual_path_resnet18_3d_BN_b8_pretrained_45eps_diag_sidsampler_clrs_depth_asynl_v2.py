fp16 = dict(loss_scale=512.)
# model settings
model = dict(
    type='DPImageClassifier',
    pretrained='/home/majiechao/pre-train/res3d18_imagenet_BN_cos_smoo_shuf_73.17-58ddff49.pth',
    aux_pretrained = './resnet18_3d_BN_b32_pretrained_45eps_lesion_sidsampler_clrs_depth_v2_aspretrain-dd049090.pth',
    backbone=dict(
        type='ResNet3D',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        depth_stride=True,      # consistent with ACS
        stem_depth_stride=False,
        stem_stride=True,
        strides=(1, 2, 2, 2),   # consistent with ACS
        in_channels=1,
        conv_cfg=dict(type='Conv3d'),
        #norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        norm_cfg = dict(type='SyncBN', requires_grad=True),
        #norm_cfg=dict(type='BN3d'),
        style='pytorch'),
    aux_backbone=dict(
        type='ResNet3D',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        depth_stride=True,      # consistent with ACS
        stem_depth_stride=False,
        stem_stride=True,
        strides=(1, 2, 2, 2),   # consistent with ACS
        in_channels=1,
        conv_cfg=dict(type='Conv3d'),
        frozen_stages=4,
        norm_eval=True,
        #norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        #norm_cfg = dict(type='SyncBN', requires_grad=True),
        norm_cfg=dict(type='BN3d'),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling', use_3d_gap=True),
    aux_neck=dict(
        type='NonLocalFusion'),
    head=dict(
        type='LinearClsHead',
        num_classes=8,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # topk=(1, 2),
        multi_cls=True,
    ))
# dataset settings
dataset_type = 'huaxiCT8Dataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    #mean=[114.495]*3, std=[57.63]*3, to_rgb=True)

train_pipeline = [
    # 1. random crop 2. rotation 3. reflection(flip by 3 axis)
    # If use transpose, transpose (d, h, w) to (h, w, d)
    dict(type='LoadTensorFromFile', data_keys='data', transpose=True, to_float32=True),
    dict(
        type='PhotoMetricDistortionMultipleSlices',
        brightness_delta=32,
        contrast_range=(0.8, 1.2)),
    dict(type='TensorNormCropFlip', crop_size=(224, 224, 64), move=16, train=True, mean=img_norm_cfg['mean'][0], std=img_norm_cfg['std'][0]),
    #dict(type='RandomResizedCrop', size=224),
    #dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    #dict(type='Normalize', **img_norm_cfg),
    dict(type='ToTensor', keys=['img'], transpose=True),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadTensorFromFile', data_keys='data', transpose=True, to_float32=True),
    dict(type='TensorNormCrop', crop_size=(224, 224, 64), move=16, train=False, mean=img_norm_cfg['mean'][0], std=img_norm_cfg['std'][0]),
    #dict(type='Resize', size=(256, -1)),
    #dict(type='CenterCrop', crop_size=224),
    #dict(type='Normalize', **img_norm_cfg),
    dict(type='ToTensor', keys=['img'],transpose=True),
    dict(type='Collect', keys=['img'])
]

root_dir = './data/CT/'

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix=root_dir + 'ct_data',
        ann_file= root_dir + 'image_ai_0612_ndiag_mod/b_paths/ct_model_train_list/huaxi_ai_diag.csv',
        use_sid_sampler=True,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=root_dir + 'ct_data',
        ann_file=root_dir + 'image_ai_0612_ndiag_mod/al_paths/ct_model_train_list/huaxi_ai_diag.csv',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix=root_dir + 'ct_data',
        ann_file=root_dir + 'image_ai_0612_ndiag_mod/a_paths/ct_model_train_list/huaxi_ai_diag.csv',
        pipeline=test_pipeline))


evaluation = dict(interval=5, metric='auc_multi_cls')

# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
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
optimizer = dict(type='Adam', lr=0.0005, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy='step', step=[35, 40, 43], warmup='constant', warmup_iters=50)
runner = dict(type='EpochBasedRunner', max_epochs=45)
