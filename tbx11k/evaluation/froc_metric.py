# tbx11k/evaluation/froc_metric.py
"""FROC metric for TBX11K — ported from VinDr-CXR mmdet v2 to mmdet v3 BaseMetric."""
from typing import List, Optional, Sequence, Tuple

import numpy as np
from mmdet.registry import METRICS
from mmengine.evaluator import BaseMetric


def _bbox_overlaps(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes (xyxy format).

    Args:
        bboxes1: (N, 4)
        bboxes2: (M, 4)

    Returns:
        iou: (N, M)
    """
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    inter_x1 = np.maximum(bboxes1[:, None, 0], bboxes2[None, :, 0])
    inter_y1 = np.maximum(bboxes1[:, None, 1], bboxes2[None, :, 1])
    inter_x2 = np.minimum(bboxes1[:, None, 2], bboxes2[None, :, 2])
    inter_y2 = np.minimum(bboxes1[:, None, 3], bboxes2[None, :, 3])

    inter_w = np.maximum(inter_x2 - inter_x1, 0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h

    union = area1[:, None] + area2[None, :] - inter_area
    return inter_area / np.maximum(union, 1e-6)


def _compute_froc(
    det_results: List[dict],
    num_images: int,
    iou_thr: float = 0.4,
    fppi_thresholds: Tuple[float, ...] = (0.1, 0.2, 0.25, 0.5, 1.0),
) -> dict:
    """Compute FROC sensitivities.

    Args:
        det_results: List of dicts, one per image. Each dict has:
            'pred_bboxes'  np.ndarray (N, 4) xyxy
            'pred_scores'  np.ndarray (N,)
            'pred_labels'  np.ndarray (N,)   0-based class idx
            'gt_bboxes'    np.ndarray (M, 4) xyxy
            'gt_labels'    np.ndarray (M,)   0-based class idx
        num_images: total number of images (for FP/image denominator).
        iou_thr: IoU threshold for TP matching.
        fppi_thresholds: FPPI operating points to report.

    Returns:
        dict with FROC@{thr} keys and mFROC.
    """
    num_classes = max(
        (int(r["pred_labels"].max()) + 1 if len(r["pred_labels"]) else 0 for r in det_results),
        default=1,
    )
    num_classes = max(
        num_classes,
        max(
            (int(r["gt_labels"].max()) + 1 if len(r["gt_labels"]) else 0 for r in det_results),
            default=1,
        ),
    )

    per_class_sens: dict = {}

    for cls_id in range(num_classes):
        all_dets: List[Tuple[float, bool]] = []
        total_gt = 0

        for img_res in det_results:
            pred_mask = img_res["pred_labels"] == cls_id
            gt_mask = img_res["gt_labels"] == cls_id

            pred_bboxes = img_res["pred_bboxes"][pred_mask]
            pred_scores = img_res["pred_scores"][pred_mask]
            gt_bboxes = img_res["gt_bboxes"][gt_mask]
            total_gt += len(gt_bboxes)

            if len(pred_bboxes) == 0:
                continue

            if len(gt_bboxes) == 0:
                for s in pred_scores:
                    all_dets.append((float(s), False))
                continue

            ious = _bbox_overlaps(pred_bboxes, gt_bboxes)
            gt_matched = np.zeros(len(gt_bboxes), dtype=bool)

            for det_idx in np.argsort(-pred_scores):
                best_gt = ious[det_idx].argmax()
                if ious[det_idx, best_gt] >= iou_thr and not gt_matched[best_gt]:
                    all_dets.append((float(pred_scores[det_idx]), True))
                    gt_matched[best_gt] = True
                else:
                    all_dets.append((float(pred_scores[det_idx]), False))

        if total_gt == 0:
            per_class_sens[cls_id] = [0.0] * len(fppi_thresholds)
            continue

        all_dets.sort(key=lambda x: -x[0])
        cum_tp = cum_fp = 0
        fppi_list, sens_list = [], []
        for _, is_tp in all_dets:
            if is_tp:
                cum_tp += 1
            else:
                cum_fp += 1
            fppi_list.append(cum_fp / num_images)
            sens_list.append(cum_tp / total_gt)

        fppi_arr = np.array(fppi_list)
        sens_arr = np.array(sens_list)

        cls_sens = []
        for thr in fppi_thresholds:
            idx = np.searchsorted(fppi_arr, thr, side="right")
            cls_sens.append(float(sens_arr[idx - 1]) if idx > 0 else 0.0)
        per_class_sens[cls_id] = cls_sens

    if not per_class_sens:
        mean_sens = [0.0] * len(fppi_thresholds)
    else:
        mean_sens = np.mean(list(per_class_sens.values()), axis=0).tolist()

    result: dict = {}
    for i, thr in enumerate(fppi_thresholds):
        result[f"FROC@{thr}"] = round(mean_sens[i], 4)
    result["mFROC"] = round(float(np.mean(mean_sens)), 4)
    return result


@METRICS.register_module()
class FROCMetric(BaseMetric):
    """FROC evaluation metric for medical lesion detection.

    Computes sensitivity at specified False Positives Per Image (FPPI)
    thresholds. Matches the standard LUNA16 / VinDr-CXR evaluation protocol.

    Args:
        iou_thr (float): IoU threshold for matching detections to GT boxes.
            Defaults to 0.4 (consistent with mAP@40).
        fppi_thresholds (tuple[float]): FPPI operating points to report.
        collect_device (str): Device for result collection in distributed eval.
        prefix (str): Prefix for metric keys in output dict.
    """

    default_prefix = "froc"

    def __init__(
        self,
        iou_thr: float = 0.4,
        fppi_thresholds: Tuple[float, ...] = (0.1, 0.2, 0.25, 0.5, 1.0),
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thr = iou_thr
        self.fppi_thresholds = tuple(fppi_thresholds)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Accumulate per-image detection and GT data."""
        for data_sample in data_samples:
            pred = data_sample["pred_instances"]
            gt = data_sample["gt_instances"]
            self.results.append(
                dict(
                    pred_bboxes=pred["bboxes"].cpu().numpy(),
                    pred_scores=pred["scores"].cpu().numpy(),
                    pred_labels=pred["labels"].cpu().numpy(),
                    gt_bboxes=gt["bboxes"].cpu().numpy(),
                    gt_labels=gt["labels"].cpu().numpy(),
                )
            )

    def compute_metrics(self, results: list) -> dict:
        """Compute FROC from accumulated per-image results."""
        return _compute_froc(
            det_results=results,
            num_images=len(results),
            iou_thr=self.iou_thr,
            fppi_thresholds=self.fppi_thresholds,
        )
