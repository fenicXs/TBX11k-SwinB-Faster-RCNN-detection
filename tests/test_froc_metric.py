# tests/test_froc_metric.py
import sys
import numpy as np
sys.path.insert(0, "/scratch/pkrish52/TBX 11/tbx11k-detection")
import tbx11k  # trigger registration

from mmdet.registry import METRICS


def _make_data_sample(pred_bboxes, pred_scores, pred_labels, gt_bboxes, gt_labels):
    """Build a minimal data_sample dict matching mmdet v3 DetDataSample structure."""
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample

    sample = DetDataSample()

    pred = InstanceData()
    pred.bboxes = pred_bboxes
    pred.scores = pred_scores
    pred.labels = pred_labels
    sample.pred_instances = pred

    gt = InstanceData()
    gt.bboxes = gt_bboxes
    gt.labels = gt_labels
    sample.gt_instances = gt

    return sample


def test_froc_metric_registered():
    assert METRICS.get("FROCMetric") is not None


def test_perfect_detection():
    """One image, one GT box, one perfect detection → sensitivity = 1.0 at all FPPI."""
    import torch
    metric_cls = METRICS.get("FROCMetric")
    metric = metric_cls(iou_thr=0.4, fppi_thresholds=(0.1, 0.2, 0.25, 0.5, 1.0))
    metric.dataset_meta = {"classes": ("tb",)}

    gt_box = torch.tensor([[10., 10., 100., 100.]])
    sample = _make_data_sample(
        pred_bboxes=torch.tensor([[10., 10., 100., 100.]]),
        pred_scores=torch.tensor([0.99]),
        pred_labels=torch.tensor([0]),
        gt_bboxes=gt_box,
        gt_labels=torch.tensor([0]),
    )
    # Convert DetDataSample to dict as mmdet evaluator does
    data_sample_dict = {
        "pred_instances": {
            "bboxes": sample.pred_instances.bboxes,
            "scores": sample.pred_instances.scores,
            "labels": sample.pred_instances.labels,
        },
        "gt_instances": {
            "bboxes": sample.gt_instances.bboxes,
            "labels": sample.gt_instances.labels,
        },
    }
    metric.process({}, [data_sample_dict])
    result = metric.compute_metrics(metric.results)

    assert "FROC@0.1" in result
    assert "mFROC" in result
    assert result["FROC@1.0"] == 1.0


def test_zero_detection():
    """No detections → sensitivity = 0.0 everywhere."""
    import torch
    metric_cls = METRICS.get("FROCMetric")
    metric = metric_cls(iou_thr=0.4, fppi_thresholds=(0.1, 0.2, 0.25, 0.5, 1.0))
    metric.dataset_meta = {"classes": ("tb",)}

    data_sample_dict = {
        "pred_instances": {
            "bboxes": torch.zeros((0, 4)),
            "scores": torch.zeros(0),
            "labels": torch.zeros(0, dtype=torch.long),
        },
        "gt_instances": {
            "bboxes": torch.tensor([[10., 10., 100., 100.]]),
            "labels": torch.tensor([0]),
        },
    }
    metric.process({}, [data_sample_dict])
    result = metric.compute_metrics(metric.results)
    assert result["mFROC"] == 0.0
