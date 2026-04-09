# tools/train.py
import os
import subprocess
import sys
from pathlib import Path

MMDET_TRAIN = "/scratch/pkrish52/TBX 11/mmdetection/tools/train.py"
REPO_ROOT = str(Path(__file__).resolve().parent.parent)


def main() -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{existing}" if existing else REPO_ROOT
    result = subprocess.run(
        [sys.executable, MMDET_TRAIN] + sys.argv[1:],
        env=env,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
