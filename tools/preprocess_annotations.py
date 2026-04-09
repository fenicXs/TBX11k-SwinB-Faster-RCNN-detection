"""Remap TBX11K 3-class COCO JSON to single 'tb' class.

Usage:
    python tools/preprocess_annotations.py \
        --train-json data/annotations/json/TBX11K_train.json \
        --val-json   data/annotations/json/TBX11K_val.json \
        --out-dir    data/annotations
"""
import argparse
import json
from pathlib import Path


SINGLE_CATEGORY = [{"id": 1, "name": "tb", "supercategory": "tb"}]


def remap(src_path: Path, dst_path: Path) -> None:
    with open(src_path) as f:
        data = json.load(f)

    data["categories"] = SINGLE_CATEGORY
    for ann in data["annotations"]:
        ann["category_id"] = 1

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(data, f)
    print(f"Written {len(data['images'])} images, "
          f"{len(data['annotations'])} annotations → {dst_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", required=True, type=Path)
    parser.add_argument("--val-json",   required=True, type=Path)
    parser.add_argument("--out-dir",    required=True, type=Path)
    args = parser.parse_args()

    remap(args.train_json, args.out_dir / "tbx11k_train_1cls.json")
    remap(args.val_json,   args.out_dir / "tbx11k_val_1cls.json")


if __name__ == "__main__":
    main()
