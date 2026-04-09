"""
Visualize model predictions vs ground truth on TBX11K val images.

Usage:
    python tools/visualize_predictions.py \
        --config configs/faster_rcnn_swinb_fpn_tbx11k.py \
        --checkpoint work_dirs/faster_rcnn_swinb_fpn_tbx11k/best_coco_mAP_40_epoch_12.pth \
        --out-dir work_dirs/faster_rcnn_swinb_fpn_tbx11k/visualizations \
        --score-thr 0.3 \
        --num-images 20
"""

import argparse
import os
import sys
import json
import random
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# ── make tbx11k importable ──────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ["PYTHONPATH"] = REPO_ROOT + ":" + os.environ.get("PYTHONPATH", "")

import tbx11k  # noqa: F401 — triggers registry


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="configs/faster_rcnn_swinb_fpn_tbx11k.py")
    p.add_argument("--checkpoint",  default="work_dirs/faster_rcnn_swinb_fpn_tbx11k/best_coco_mAP_40_epoch_12.pth")
    p.add_argument("--ann-file",    default="/scratch/pkrish52/TBX 11/TBX11K/annotations/json/TBX11K_val.json")
    p.add_argument("--img-root",    default="/scratch/pkrish52/TBX 11/TBX11K/imgs")
    p.add_argument("--out-dir",     default="work_dirs/faster_rcnn_swinb_fpn_tbx11k/visualizations")
    p.add_argument("--score-thr",   type=float, default=0.3)
    p.add_argument("--num-images",  type=int,   default=20)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--include-negatives", action="store_true",
                   help="Also include some non-TB images (false positive check)")
    return p.parse_args()


def load_model(config_path, checkpoint_path):
    from mmdet.apis import init_detector
    model = init_detector(config_path, checkpoint_path, device="cuda:0")
    return model


def run_inference(model, img_path, score_thr):
    from mmdet.apis import inference_detector
    result = inference_detector(model, img_path)
    # result.pred_instances contains bboxes, scores, labels
    inst = result.pred_instances
    bboxes = inst.bboxes.cpu().numpy()   # (N, 4) xyxy
    scores = inst.scores.cpu().numpy()   # (N,)
    keep   = scores >= score_thr
    return bboxes[keep], scores[keep]


def draw_single(ax, img_path, gt_boxes_xywh, pred_boxes_xyxy, pred_scores, score_thr, img_id):
    """Draw one CXR with GT (green) and predictions (red) overlaid."""
    img = np.array(Image.open(img_path).convert("RGB"))
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax.axis("off")

    has_gt   = len(gt_boxes_xywh) > 0
    has_pred = len(pred_boxes_xyxy) > 0

    # Ground truth — green dashed
    for (x, y, w, h) in gt_boxes_xywh:
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor="#00FF00", facecolor="none",
            linestyle="--", label="GT"
        )
        ax.add_patch(rect)

    # Predictions — red solid
    for (x1, y1, x2, y2), score in zip(pred_boxes_xyxy, pred_scores):
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="#FF4444", facecolor="none",
            linestyle="-", label="Pred"
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, f"{score:.2f}",
                color="#FF4444", fontsize=7, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))

    status = []
    if has_gt and has_pred:     status_str, color = "TP",   "#00CC44"
    elif has_gt and not has_pred: status_str, color = "FN", "#FF8800"
    elif not has_gt and has_pred: status_str, color = "FP", "#FF2222"
    else:                         status_str, color = "TN", "#888888"

    fname = os.path.basename(img_path)
    ax.set_title(f"{fname}\n{status_str} | GT:{len(gt_boxes_xywh)} Pred:{len(pred_boxes_xyxy)}",
                 fontsize=8, color=color, pad=3)


def build_legend(fig):
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#00FF00", linewidth=2, linestyle="--", label="Ground Truth"),
        Line2D([0], [0], color="#FF4444", linewidth=2, linestyle="-",  label="Prediction"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=10,
               framealpha=0.8, bbox_to_anchor=(0.5, 0.01))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    # ── Load annotations ────────────────────────────────────────────────────
    with open(args.ann_file) as f:
        coco = json.load(f)

    # Build lookup: image_id → list of [x,y,w,h] GT boxes
    gt_map = {}
    for ann in coco["annotations"]:
        gt_map.setdefault(ann["image_id"], []).append(ann["bbox"])

    # Separate TB and non-TB val images
    all_imgs   = coco["images"]
    tb_imgs    = [img for img in all_imgs if img["id"] in gt_map]
    nontb_imgs = [img for img in all_imgs if img["id"] not in gt_map]

    # Sample images
    n_tb = args.num_images if not args.include_negatives else int(args.num_images * 0.75)
    n_tb = min(n_tb, len(tb_imgs))
    selected = random.sample(tb_imgs, n_tb)

    if args.include_negatives:
        n_neg = args.num_images - n_tb
        selected += random.sample(nontb_imgs, min(n_neg, len(nontb_imgs)))
        random.shuffle(selected)

    print(f"Selected {len(selected)} images ({n_tb} TB + {len(selected)-n_tb} non-TB)")

    # ── Load model ──────────────────────────────────────────────────────────
    config_path = os.path.join(REPO_ROOT, args.config)
    ckpt_path   = os.path.join(REPO_ROOT, args.checkpoint)
    print(f"Loading model from {ckpt_path} ...")
    model = load_model(config_path, ckpt_path)
    print("Model loaded.")

    # ── Run inference + visualize ───────────────────────────────────────────
    cols = 4
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4 + 0.5))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    tp = fn = fp = tn = 0

    for i, img_meta in enumerate(selected):
        img_path = os.path.join(args.img_root, img_meta["file_name"])
        gt_boxes = gt_map.get(img_meta["id"], [])

        pred_bboxes, pred_scores = run_inference(model, img_path, args.score_thr)

        draw_single(axes[i], img_path, gt_boxes, pred_bboxes, pred_scores,
                    args.score_thr, img_meta["id"])

        has_gt   = len(gt_boxes) > 0
        has_pred = len(pred_bboxes) > 0
        if   has_gt and has_pred:       tp += 1
        elif has_gt and not has_pred:   fn += 1
        elif not has_gt and has_pred:   fp += 1
        else:                           tn += 1

    # Hide unused axes
    for j in range(len(selected), len(axes)):
        axes[j].set_visible(False)

    build_legend(fig)
    fig.suptitle(
        f"TBX11K Val — Faster-RCNN Swin-B  |  Score thr={args.score_thr}  |  "
        f"TP={tp}  FN={fn}  FP={fp}  TN={tn}",
        fontsize=12, y=0.995
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.995])

    out_path = os.path.join(args.out_dir, f"predictions_thr{args.score_thr:.2f}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")
    print(f"Summary over {len(selected)} sampled images:")
    print(f"  TP (GT + Pred)    : {tp}")
    print(f"  FN (GT, no Pred)  : {fn}")
    print(f"  FP (no GT, Pred)  : {fp}")
    print(f"  TN (no GT, no Pred): {tn}")

    # ── Also save individual full-res images ─────────────────────────────────
    ind_dir = os.path.join(args.out_dir, "individual")
    os.makedirs(ind_dir, exist_ok=True)
    print(f"\nSaving {len(selected)} individual images to {ind_dir}/ ...")
    for img_meta in selected:
        img_path = os.path.join(args.img_root, img_meta["file_name"])
        gt_boxes = gt_map.get(img_meta["id"], [])
        pred_bboxes, pred_scores = run_inference(model, img_path, args.score_thr)

        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
        draw_single(ax2, img_path, gt_boxes, pred_bboxes, pred_scores,
                    args.score_thr, img_meta["id"])
        build_legend(fig2)
        plt.tight_layout()
        fname = os.path.splitext(os.path.basename(img_meta["file_name"]))[0]
        fig2.savefig(os.path.join(ind_dir, f"{fname}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig2)
    print("Done.")


if __name__ == "__main__":
    main()
