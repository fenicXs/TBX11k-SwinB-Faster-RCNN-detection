# configs/faster_rcnn_swinb_fpn_tbx11k.py
_base_ = [
    "/scratch/pkrish52/TBX 11/mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py",
    "_base_/tbx11k_dataset.py",
    "_base_/schedule_12e.py",
    "_base_/runtime.py",
]

model = dict(
    backbone=dict(
        _delete_=True,
        type="SwinTransformer",
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="/scratch/pkrish52/swin_base_patch4_window7_224.pth",
        ),
    ),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    roi_head=dict(bbox_head=dict(num_classes=1)),
)

work_dir = "/scratch/pkrish52/TBX 11/tbx11k-detection/work_dirs/faster_rcnn_swinb_fpn_tbx11k"
