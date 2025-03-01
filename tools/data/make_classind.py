import sys

sys.path.append(".")

from pathlib import Path

from assertpy.assertpy import assert_that
from config import settings as conf

ROOT = Path.cwd()
DATASET = conf.active.dataset
DATASET_DIR = ROOT / conf[DATASET].path
OUT_FILE = DATASET_DIR.parent / "annotations/classInd.txt"

assert_that(DATASET_DIR).is_directory().exists()

subdirs = [subdir.name for subdir in DATASET_DIR.iterdir() if subdir.is_dir()]

subdirs.sort()
OUT_FILE.parent.mkdir(exist_ok=True, parents=True)

with open(OUT_FILE, "w") as f:
    for i, subdir in enumerate(subdirs, start=1):
        f.write(f"{i} {subdir}\n")
