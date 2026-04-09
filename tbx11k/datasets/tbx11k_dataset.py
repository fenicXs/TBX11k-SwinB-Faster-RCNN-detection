# tbx11k/datasets/tbx11k_dataset.py
from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class TBX11KDataset(CocoDataset):
    """TBX11K tuberculosis detection dataset.

    Single class 'tb' — all ActiveTuberculosis, ObsoletePulmonaryTuberculosis,
    and PulmonaryTuberculosis annotations collapsed to one category.
    """

    METAINFO = {
        "classes": ("tb",),
        "palette": [(220, 20, 60)],
    }
