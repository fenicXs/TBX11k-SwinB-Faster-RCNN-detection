import json, subprocess, sys, os, tempfile, shutil
from pathlib import Path

REPO = Path("/scratch/pkrish52/TBX 11/tbx11k-detection")
TRAIN_JSON = REPO / "data/annotations/json/TBX11K_train.json"
VAL_JSON   = REPO / "data/annotations/json/TBX11K_val.json"

def test_output_has_single_category():
    out_dir = Path(tempfile.mkdtemp())
    try:
        subprocess.run(
            [sys.executable, str(REPO / "tools/preprocess_annotations.py"),
             "--train-json", str(TRAIN_JSON),
             "--val-json", str(VAL_JSON),
             "--out-dir", str(out_dir)],
            check=True
        )
        for fname in ["tbx11k_train_1cls.json", "tbx11k_val_1cls.json"]:
            with open(out_dir / fname) as f:
                data = json.load(f)
            assert data["categories"] == [{"id": 1, "name": "tb", "supercategory": "tb"}]
            for ann in data["annotations"]:
                assert ann["category_id"] == 1
    finally:
        shutil.rmtree(out_dir)

def test_image_and_annotation_counts_preserved():
    out_dir = Path(tempfile.mkdtemp())
    try:
        subprocess.run(
            [sys.executable, str(REPO / "tools/preprocess_annotations.py"),
             "--train-json", str(TRAIN_JSON),
             "--val-json", str(VAL_JSON),
             "--out-dir", str(out_dir)],
            check=True
        )
        with open(TRAIN_JSON) as f:
            orig = json.load(f)
        with open(out_dir / "tbx11k_train_1cls.json") as f:
            out = json.load(f)
        assert len(out["images"]) == len(orig["images"])
        assert len(out["annotations"]) == len(orig["annotations"])
    finally:
        shutil.rmtree(out_dir)
