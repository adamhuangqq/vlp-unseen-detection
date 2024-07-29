_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py', 'mmdet::_base_/default_runtime.py',
    'mixpl_coco_detection.py'
]
detector = _base_.model
detector.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    bgr_to_rgb=False,
    pad_size_divisor=32)
detector.backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=False),
    norm_eval=True,
    style='caffe',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='open-mmlab://detectron2/resnet50_caffe'))


model = dict(
    type='SemiBaseDetector',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector['data_preprocessor']),
    semi_train_cfg=dict(
        least_num=1,
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=1.0,
        cls_pseudo_thr=0.7,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))


# 10% coco train2017 is set as labeled dataset
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset
labeled_dataset.ann_file = 'annotations/instances_train2017.json'
unlabeled_dataset.ann_file = 'annotations/instances_val2017.json'
# labeled_dataset.ann_file = 'semi_anns/instances_train2017.1@50.json'
# unlabeled_dataset.ann_file = 'semi_anns/instances_val+unlabeled.json'
labeled_dataset.data_prefix = dict(img='train2017/')
unlabeled_dataset.data_prefix = dict(img='train2017/')

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    sampler=dict(batch_size=8, source_ratio=[4, 4]),
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))

# training schedule for 180k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=90000, val_interval=3000)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=5000,
        end=90000,
        by_epoch=False,
        milestones=[10000, 40000, 80000],
        gamma=0.5)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=20, norm_type=2)
)

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=500, max_keep_ckpts=2))
log_processor = dict(by_epoch=False)
custom_hooks = [dict(type='MeanTeacherHook', momentum=0.0002, gamma=4)]
resume=True
