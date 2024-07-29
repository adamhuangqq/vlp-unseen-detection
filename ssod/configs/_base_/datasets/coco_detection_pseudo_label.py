# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

backend_args = None
# metainfo = {
#     'classes':
#     ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
#         'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
#         'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
#     # palette is a list of color tuples, which is used for visualization.
#     'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
#                 (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
#                 (153, 69, 1), (120, 166, 157), (0, 182, 199),
#                 (0, 226, 252), (182, 182, 255), (0, 0, 230), (220, 20, 60),
#                 (163, 255, 0), (0, 82, 0), (3, 95, 161), (0, 80, 100),
#                 (183, 130, 88)]
# }

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
label_dataset=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='class_od/annotations/train.json',
    data_prefix=dict(img='train2017/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)

pseudo_dataset=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='class_od/pseudo_label/train_pl_fcos_50p_s0.3.json',
    data_prefix=dict(img='train2017/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)

batch_size=8
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='ConcatDataset', datasets=[label_dataset, pseudo_dataset])
    )

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='class_od/annotations/test_dataset_od.json.json',
        data_prefix=dict(img='train2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'class_od/annotations/test_dataset_od.json.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     metric='bbox',
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_detection/test')
