# configs/_base_/tbx11k_dataset.py
dataset_type = "TBX11KDataset"
data_root = "/scratch/pkrish52/TBX 11/tbx11k-detection/data/"

backend_args = None

# Multi-scale training augmentation
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="RandomChoiceResize",
        scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                (736, 1333), (768, 1333), (800, 1333)],
        keep_ratio=True,
    ),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

val_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(800, 1333), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/tbx11k_train_1cls.json",
        data_prefix=dict(img="imgs/"),
        filter_cfg=dict(filter_empty_gt=False),   # include health/sick images
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/tbx11k_val_1cls.json",
        data_prefix=dict(img="imgs/"),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type="TBX11KCocoMetric",
        ann_file=data_root + "annotations/tbx11k_val_1cls.json",
        metric="bbox",
        classwise=True,
        backend_args=backend_args,
    ),
    dict(
        type="FROCMetric",
        iou_thr=0.4,
        fppi_thresholds=(0.1, 0.2, 0.25, 0.5, 1.0),
    ),
]

test_evaluator = val_evaluator
