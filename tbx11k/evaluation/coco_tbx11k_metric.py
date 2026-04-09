# tbx11k/evaluation/coco_tbx11k_metric.py
"""Extended CocoMetric reporting mAP@40, mAP@50, and mAP@50:95."""
from typing import Dict, List

import numpy as np
from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS


@METRICS.register_module()
class TBX11KCocoMetric(CocoMetric):
    """CocoMetric subclass that reports mAP@40, mAP@50, and mAP@50:95.

    Runs three COCO evaluations with different IoU thresholds and reports
    each result individually.

    The primary metric key for checkpoint selection is ``mAP_40``.

    Args:
        ann_file (str): Path to COCO-format annotation JSON.
        metric (str): Metric type. Only 'bbox' supported here.
        classwise (bool): Whether to report per-class AP.
        **kwargs: Passed to parent CocoMetric.
    """

    # Maps friendly key → iou_thrs argument for CocoMetric
    iou_thrs_map: Dict[str, List[float]] = {
        "mAP_40": [0.4],
        "mAP_50": [0.5],
        "mAP_50_95": np.linspace(0.5, 0.95, 10).tolist(),
    }

    def compute_metrics(self, results: list) -> dict:
        """Run COCO evaluation at three IoU thresholds and merge results.

        The parent CocoMetric's ``classwise`` logic assumes the standard COCO
        10-IoU grid (0.5:0.95) and hard-codes IoU indices 0 and 5 when
        building per-category tables.  When we override ``iou_thrs`` to a
        single threshold (e.g. [0.4] or [0.5]) the precisions tensor has only
        one entry along the IoU axis, so index 5 is out of bounds.

        We therefore disable ``classwise`` for all but the 50:95 evaluation
        (which uses the standard 10-IoU grid and is safe) to avoid the crash.
        """
        all_metrics: dict = {}

        for key, iou_thrs in self.iou_thrs_map.items():
            orig_iou = self.iou_thrs
            # Only the 10-IoU evaluation is safe with classwise=True
            orig_classwise = self.classwise
            if key != "mAP_50_95":
                self.classwise = False

            self.iou_thrs = iou_thrs
            try:
                metrics = super().compute_metrics(results)
            finally:
                self.iou_thrs = orig_iou
                self.classwise = orig_classwise

            # parent returns 'bbox_mAP' — rename to our key
            if "bbox_mAP" in metrics:
                all_metrics[key] = metrics.pop("bbox_mAP")
            # carry per-class APs with prefixed key
            for mk, mv in metrics.items():
                all_metrics[f"{key}_{mk}"] = mv

        return all_metrics
