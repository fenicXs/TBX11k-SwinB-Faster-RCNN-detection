# tests/test_dataset.py
import sys
sys.path.insert(0, "/scratch/pkrish52/TBX 11/tbx11k-detection")
import tbx11k  # trigger registration

from mmdet.registry import DATASETS

def test_dataset_registered():
    assert DATASETS.get("TBX11KDataset") is not None

def test_metainfo():
    cls = DATASETS.get("TBX11KDataset")
    assert cls.METAINFO["classes"] == ("tb",)
    assert len(cls.METAINFO["palette"]) == 1
