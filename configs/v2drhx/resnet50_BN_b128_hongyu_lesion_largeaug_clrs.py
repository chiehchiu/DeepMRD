fp16 = dict(loss_scale=512.)
# model settings
model = dict(
    type='ImageClassifier',
    pretrained='resnet50.pt',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=18,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        multi_cls=True,
    ))
# dataset settings
dataset_type = 'huaxiDRlesion18Dataset'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='XrayTrain', transCrop=224, scale=(0.09, 1)),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='XrayTest', transResize=256, transCrop=224),
    dict(type='Collect', keys=['img'])
]

root_dir = './data/CR/'

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix=root_dir + 'image1024',
        ann_file= root_dir + '0528-1-mod_list/b_paths/dr_model_train_list/huaxi_ai_lesion.csv',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=root_dir + 'image1024',
        ann_file=root_dir + '0528-1-mod_list/a_paths/dr_model_train_list/huaxi_ai_lesion.csv',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix=root_dir + 'image1024',
        ann_file=root_dir + 'image_ai_ctcrs0618/b_paths/dr_model_train_list/huaxi_ai_lesion.csv',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='auc_multi_cls')

# optimizer
optimizer = dict(type='Adam', lr=1e-4, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[25, 35])
runner = dict(type='EpochBasedRunner', max_epochs=45)

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
