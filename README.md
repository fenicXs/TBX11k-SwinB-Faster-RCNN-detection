# TBX11K Detection — Swin-B + Faster-RCNN

TB lesion detection on the [TBX11K](https://www.kaggle.com/datasets/usmanshams/tbx-11) dataset using a **Swin-B backbone + FPN + Faster-RCNN** head, built with [MMDetection 3.3.0](https://github.com/open-mmlab/mmdetection).

---

## Results

**Val set** (1,800 images, 309 GT boxes, epoch 12 / best checkpoint):

| Metric | Score |
|--------|-------|
| **mAP@40** | **0.838** |
| mAP@50 | 0.778 |
| mAP@50:95 | 0.353 |
| mFROC | 0.893 |

**FROC operating points:**

| FP/image | Sensitivity |
|----------|------------|
| 0.10 | 0.851 |
| 0.20 | 0.884 |
| 0.25 | 0.896 |
| 0.50 | 0.916 |
| 1.00 | 0.916 |

---

## Dataset

**TBX11K** — 11,200 chest X-rays (512×512 PNG) with COCO-format bounding-box annotations for TB lesions.

| Split | Images | TB boxes |
|-------|--------|----------|
| Train | 6,600  | 902 |
| Val   | 1,800  | 309 |
| Test  | 2,800  | 0 (withheld) |

Three category IDs (`ActiveTuberculosis`, `ObsoletePulmonaryTuberculosis`, `PulmonaryTuberculosis`) are collapsed to a single `tb` class. Non-TB images (healthy + sick-non-TB, ~5,800 of 6,600 train) are kept as hard negatives (`filter_empty_gt=False`).

Download the dataset from Kaggle and place it at:
```
data/
└── TBX11K/
    ├── imgs/
    │   ├── tb/
    │   ├── health/
    │   ├── sick/
    │   └── test/
    └── annotations/
        └── json/
            ├── TBX11K_train.json
            └── TBX11K_val.json
```

---

## Environment Setup

### 1. Clone this repo and MMDetection

```bash
git clone https://github.com/fenicXs/TBX11k-SwinB-Faster-RCNN-detection.git
cd TBX11k-SwinB-Faster-RCNN-detection

# MMDetection must be cloned as a sibling directory
cd ..
git clone https://github.com/open-mmlab/mmdetection.git
cd TBX11k-SwinB-Faster-RCNN-detection
```

Expected layout:
```
parent_dir/
├── TBX11k-SwinB-Faster-RCNN-detection/   ← this repo
└── mmdetection/                           ← cloned mmdetection
```

> `tools/train.py` and `tools/test.py` reference `../mmdetection` by default.

### 2. Create conda environment

```bash
conda create -n tbx11k python=3.10 -y
conda activate tbx11k
```

### 3. Install PyTorch (CUDA 12.4)

```bash
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124
```

Adjust the CUDA version to match your system.

### 4. Install MMDet stack

```bash
pip install mmengine==0.10.7 mmcv==2.1.0 mmdet==3.3.0
pip install pycocotools numpy matplotlib pillow
```

### 5. Fix PyTorch 2.6 weights_only breaking change

PyTorch 2.6 changed `torch.load()` default to `weights_only=True`, which breaks MMEngine checkpoint loading. Patch it:

```bash
python - << 'EOF'
import mmengine, pathlib, re

ckpt = pathlib.Path(mmengine.__file__).parent / "runner" / "checkpoint.py"
src  = ckpt.read_text()
if 'weights_only=False' not in src:
    patched = re.sub(
        r'(torch\.load\([^)]*\))',
        lambda m: m.group(0).rstrip(')') + ', weights_only=False)',
        src
    )
    ckpt.write_text(patched)
    print("Patched", ckpt)
else:
    print("Already patched.")
EOF
```

### 6. Download Swin-B ImageNet pretrained weights

```bash
mkdir -p pretrained
wget -O pretrained/swin_base_patch4_window7_224.pth \
    https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
```

Update the `load_from` path in `configs/faster_rcnn_swinb_fpn_tbx11k.py` if you place it elsewhere.

---

## Preprocessing

Remap 3 category IDs → single `tb` class and generate 1-class annotation JSONs:

```bash
python tools/preprocess_annotations.py \
    --train-ann data/TBX11K/annotations/json/TBX11K_train.json \
    --val-ann   data/TBX11K/annotations/json/TBX11K_val.json \
    --out-dir   data/annotations
```

Outputs:
- `data/annotations/tbx11k_train_1cls.json` — 6,600 images, 902 annotations
- `data/annotations/tbx11k_val_1cls.json` — 1,800 images, 309 annotations

---

## Training

```bash
python tools/train.py \
    configs/faster_rcnn_swinb_fpn_tbx11k.py \
    --work-dir work_dirs/faster_rcnn_swinb_fpn_tbx11k \
    --amp
```

To run in the background:
```bash
nohup python tools/train.py \
    configs/faster_rcnn_swinb_fpn_tbx11k.py \
    --work-dir work_dirs/faster_rcnn_swinb_fpn_tbx11k \
    --amp > work_dirs/faster_rcnn_swinb_fpn_tbx11k/train.log 2>&1 &
```

Training runs for **12 epochs**. The best checkpoint is saved by `coco/mAP_40`.

### Key hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 1e-4 |
| Backbone LR multiplier | 0.1 |
| Weight decay | 0.05 |
| LR decay epochs | 8, 11 |
| Batch size | 2 per GPU |
| Multi-scale aug | 11 discrete scales [480–800], max_size=1333 |
| Normalization | ImageNet mean/std |

---

## Evaluation

```bash
python tools/test.py \
    configs/faster_rcnn_swinb_fpn_tbx11k.py \
    work_dirs/faster_rcnn_swinb_fpn_tbx11k/best_coco_mAP_40_epoch_12.pth
```

Reports:
- `coco/mAP_40`, `coco/mAP_50`, `coco/mAP_50_95`
- `froc/FROC@{0.1,0.2,0.25,0.5,1.0}`, `froc/mFROC`

---

## Live Training Dashboard

Monitor training progress in real time (refreshes every 60 s):

```bash
python tools/plot_live.py \
    --work-dir work_dirs/faster_rcnn_swinb_fpn_tbx11k \
    --interval 60
```

Saves `{work_dir}/plots/dashboard.png` with three panels:
1. Training loss curves (all 5 components, EMA-smoothed)
2. Val metrics per epoch — mAP@40/50/50:95 and mFROC
3. FROC curves — one per epoch, faint→dark blue = early→latest

---

## Prediction Visualization

Overlay predictions and ground truth on val images:

```bash
python tools/visualize_predictions.py \
    --config configs/faster_rcnn_swinb_fpn_tbx11k.py \
    --checkpoint work_dirs/faster_rcnn_swinb_fpn_tbx11k/best_coco_mAP_40_epoch_12.pth \
    --score-thr 0.3 \
    --num-images 20 \
    --out-dir work_dirs/faster_rcnn_swinb_fpn_tbx11k/visualizations
```

Add `--include-negatives` to also sample non-TB images (false positive check).

---

## Repository Structure

```
TBX11k-SwinB-Faster-RCNN-detection/
├── configs/
│   ├── _base_/
│   │   ├── tbx11k_dataset.py      # data pipeline, augmentation, loaders
│   │   ├── schedule_12e.py        # AdamW optimizer + LR schedule
│   │   └── runtime.py             # hooks, logging, custom_imports
│   └── faster_rcnn_swinb_fpn_tbx11k.py  # full model config
├── tbx11k/
│   ├── __init__.py
│   ├── datasets/
│   │   └── tbx11k_dataset.py      # CocoDataset subclass (1-class)
│   └── evaluation/
│       ├── coco_tbx11k_metric.py  # mAP@40/50/50:95 evaluator
│       └── froc_metric.py         # FROC @ multiple FP/img thresholds
├── tools/
│   ├── train.py                   # training entry point
│   ├── test.py                    # evaluation entry point
│   ├── preprocess_annotations.py  # 3-class → 1-class JSON remapping
│   ├── plot_live.py               # live dashboard (polls scalars.json)
│   └── visualize_predictions.py   # pred + GT overlay on val images
├── tests/                         # pytest unit tests
├── requirements.txt
└── README.md
```

---

## Citation

If you use this codebase, please cite the TBX11K dataset:

```bibtex
@inproceedings{liu2020rethinking,
  title={Rethinking Computer-Aided Tuberculosis Diagnosis},
  author={Liu, Yun and Wu, Yu-Huan and Ban, Yunfeng and Wang, Huifang and Cheng, Ming-Ming},
  booktitle={CVPR},
  year={2020}
}
```
