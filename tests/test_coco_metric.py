# tests/test_coco_metric.py
import sys
sys.path.insert(0, "/scratch/pkrish52/TBX 11/tbx11k-detection")
import tbx11k  # trigger registration

from mmdet.registry import METRICS


def test_metric_registered():
    assert METRICS.get("TBX11KCocoMetric") is not None


def test_iou_thrs_map_keys():
    """Metric must define iou_thrs_map with mAP_40, mAP_50, mAP_50_95 keys."""
    metric_cls = METRICS.get("TBX11KCocoMetric")
    ann_file = "/scratch/pkrish52/TBX 11/tbx11k-detection/data/annotations/tbx11k_val_1cls.json"
    metric = metric_cls(ann_file=ann_file, metric="bbox", classwise=True)
    expected_keys = {"mAP_40", "mAP_50", "mAP_50_95"}
    assert set(metric.iou_thrs_map.keys()) == expected_keys
