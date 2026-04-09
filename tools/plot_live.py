"""Live training dashboard — polls MMEngine scalars.json and saves updated plots.

Generates three plots that refresh automatically as training progresses:
  1. Training loss curve (all loss components vs iteration)
  2. Validation metric curve (mAP@40, mAP@50, mAP50:95, mFROC vs epoch)
  3. FROC curve (latest sensitivity vs FP/image operating points)

Plots are saved as PNGs in {work_dir}/plots/ — open them in VSCode and they
refresh automatically each time the script updates them.

Usage:
    # Live mode (refreshes every 60s while training):
    conda run -n mmseg_ptx python tools/plot_live.py \\
        --work-dir work_dirs/faster_rcnn_swinb_fpn_tbx11k

    # One-shot mode (generate once and exit):
    conda run -n mmseg_ptx python tools/plot_live.py \\
        --work-dir work_dirs/faster_rcnn_swinb_fpn_tbx11k --once

    # Custom refresh interval (30s):
    conda run -n mmseg_ptx python tools/plot_live.py \\
        --work-dir work_dirs/faster_rcnn_swinb_fpn_tbx11k --interval 30
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works over SSH / without display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Log parsing
# ──────────────────────────────────────────────────────────────────────────────

def find_scalars_json(work_dir: Path) -> Optional[Path]:
    """Find the most recently modified scalars.json under work_dir."""
    candidates = sorted(
        work_dir.rglob("scalars.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def parse_scalars(path: Path) -> Tuple[List[dict], List[dict]]:
    """Parse scalars.json into train and val record lists.

    Returns:
        train_records: dicts with loss keys + 'step' (global iteration)
        val_records:   dicts with metric keys + 'step' (epoch number)
    """
    train_records: List[dict] = []
    val_records: List[dict] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "loss" in rec:
                train_records.append(rec)
            elif "coco/mAP_40" in rec:
                val_records.append(rec)

    return train_records, val_records


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

LOSS_COLORS = {
    "loss":          ("#2196F3", 2.0),   # total loss — thick blue
    "loss_rpn_cls":  ("#FF9800", 1.2),
    "loss_rpn_bbox": ("#F44336", 1.2),
    "loss_cls":      ("#4CAF50", 1.2),
    "loss_bbox":     ("#9C27B0", 1.2),
}

METRIC_COLORS = {
    "coco/mAP_40":    ("#2196F3", "mAP@40 (primary)", 2.5),
    "coco/mAP_50":    ("#FF9800", "mAP@50",           1.5),
    "coco/mAP_50_95": ("#9E9E9E", "mAP@50:95",        1.5),
    "froc/mFROC":     ("#E91E63", "mFROC",            2.0),
}

FROC_THRS = [0.1, 0.2, 0.25, 0.5, 1.0]
FROC_KEYS = [f"froc/FROC@{t}" for t in FROC_THRS]


def _smooth(values: List[float], weight: float = 0.85) -> List[float]:
    """Exponential moving average smoothing (TensorBoard-style)."""
    smoothed, last = [], values[0] if values else 0.0
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return smoothed


def plot_loss(ax: plt.Axes, train: List[dict]) -> None:
    ax.clear()
    if not train:
        ax.set_title("Training Loss (waiting for data…)")
        return

    steps = [r["step"] for r in train]
    for key, (color, lw) in LOSS_COLORS.items():
        vals = [r.get(key, float("nan")) for r in train]
        if all(np.isnan(vals)):
            continue
        raw = [v for v in vals if not np.isnan(v)]
        sm = _smooth(raw)
        ax.plot(steps, vals, color=color, alpha=0.2, linewidth=0.8)
        ax.plot(steps, sm,   color=color, linewidth=lw,
                label=key.replace("loss_", "").replace("loss", "total"))

    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("Training Loss", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_metrics(ax: plt.Axes, val: List[dict]) -> None:
    ax.clear()
    if not val:
        ax.set_title("Val Metrics (waiting for epoch 1…)")
        return

    epochs = [r["step"] for r in val]
    for key, (color, label, lw) in METRIC_COLORS.items():
        vals = [r.get(key, float("nan")) for r in val]
        if all(np.isnan(vals)):
            continue
        ax.plot(epochs, vals, "o-", color=color, linewidth=lw,
                markersize=5, label=label)
        # Annotate last value
        last_v = next((v for v in reversed(vals) if not np.isnan(v)), None)
        if last_v is not None:
            ax.annotate(f"{last_v:.3f}", xy=(epochs[-1], last_v),
                        xytext=(4, 2), textcoords="offset points",
                        fontsize=7, color=color)

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Validation Metrics", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(left=0)
    if epochs:
        ax.set_xticks(range(1, max(epochs) + 1))
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_froc(ax: plt.Axes, val: List[dict]) -> None:
    ax.clear()
    if not val:
        ax.set_title("FROC Curve (waiting for epoch 1…)")
        return

    # Draw all epochs as faint grey, latest as bold color
    cmap = plt.cm.Blues
    n = len(val)

    for i, rec in enumerate(val):
        sens = [rec.get(k, float("nan")) for k in FROC_KEYS]
        if all(np.isnan(sens)):
            continue
        alpha = 0.15 + 0.6 * (i / max(n - 1, 1))
        lw    = 0.8 + 1.6 * (i / max(n - 1, 1))
        color = cmap(0.3 + 0.6 * (i / max(n - 1, 1)))
        label = f"Epoch {rec['step']}" if i == n - 1 else None
        ax.plot(FROC_THRS, sens, "o-", color=color,
                linewidth=lw, markersize=4, alpha=alpha, label=label)

    ax.set_xlabel("False Positives / Image", fontsize=10)
    ax.set_ylabel("Sensitivity", fontsize=10)
    ax.set_title("FROC Curve (latest = darkest)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(FROC_THRS)
    if val:
        ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ──────────────────────────────────────────────────────────────────────────────
# Main dashboard
# ──────────────────────────────────────────────────────────────────────────────

def render(work_dir: Path, out_dir: Path) -> str:
    """Parse latest scalars.json and save all three plots. Returns status line."""
    scalars_path = find_scalars_json(work_dir)
    if scalars_path is None:
        return "No scalars.json found yet — training hasn't started logging?"

    train, val = parse_scalars(scalars_path)

    fig = plt.figure(figsize=(18, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    ax_loss    = fig.add_subplot(gs[0])
    ax_metrics = fig.add_subplot(gs[1])
    ax_froc    = fig.add_subplot(gs[2])

    plot_loss(ax_loss, train)
    plot_metrics(ax_metrics, val)
    plot_froc(ax_froc, val)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur_epoch = val[-1]["step"] if val else 0
    cur_iter  = train[-1]["step"] if train else 0
    cur_loss  = train[-1].get("loss", float("nan")) if train else float("nan")

    fig.suptitle(
        f"TBX11K Training Dashboard  |  Epoch {cur_epoch}/12  "
        f"|  Iter {cur_iter}  |  Loss {cur_loss:.4f}  |  {ts}",
        fontsize=11, y=1.01,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = out_dir / "dashboard.png"
    fig.savefig(dashboard_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    mAP40  = val[-1].get("coco/mAP_40",  float("nan")) if val else float("nan")
    mFROC  = val[-1].get("froc/mFROC",   float("nan")) if val else float("nan")
    return (
        f"[{ts}]  ep {cur_epoch:2d}/12  iter {cur_iter:5d}  "
        f"loss {cur_loss:.4f}  mAP@40 {mAP40:.4f}  mFROC {mFROC:.4f}  "
        f"→ {dashboard_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Live training dashboard")
    parser.add_argument(
        "--work-dir", required=True, type=Path,
        help="MMDet work_dir (contains timestamped subdirs with vis_data/)",
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Refresh interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Generate plots once and exit (no polling loop)",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    if not work_dir.exists():
        sys.exit(f"work_dir does not exist: {work_dir}")

    out_dir = work_dir / "plots"

    if args.once:
        status = render(work_dir, out_dir)
        print(status)
        return

    print(f"Live dashboard — refreshing every {args.interval}s")
    print(f"Open {out_dir}/dashboard.png in VSCode to watch live")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            status = render(work_dir, out_dir)
            print(status)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
