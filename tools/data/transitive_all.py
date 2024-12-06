import sys

sys.path.append(".")

import json
from os import symlink
from pathlib import Path
from shutil import copy2

from tqdm import tqdm

from config import settings as conf

ROOT = Path.cwd()
DETECTOR = conf.active.detector
OUT_DIR = ROOT / "data/transitive"
RELEVANCY = conf.active.relevancy
datasets = "ucf101", "hmdb51", "kinetics100"
count = 0

for dataset in datasets:
    ext = conf[dataset].ext
    dataset_dir = ROOT / "data" / dataset
    acm_mask_dir = dataset_dir / DETECTOR / "0.5/actorcutmix/REPP/mask"
    icm_mask_dir = (
        dataset_dir
        / DETECTOR
        / "0.5/intercutmix/REPP/mask"
        / RELEVANCY.method
        / str(RELEVANCY.threshold)
    )

    with open(acm_mask_dir / "ratio.json") as f:
        acm_json = json.load(f)

    with open(icm_mask_dir / "ratio.json") as f:
        icm_json = json.load(f)

    stem2label = {}
    count = 0
    stem2label = {
        file.stem: file.parent.name
        for file in (dataset_dir / "videos").glob(f"**/*{ext}")
    }

    for stem in tqdm(acm_json.keys(), dynamic_ncols=True):
        acm_ratio = acm_json[stem]
        icm_ratio = icm_json[stem]

        if icm_ratio <= acm_ratio:
            continue

        label = stem2label[stem]
        src = dataset_dir / "videos" / label / (stem + ext)
        dst = OUT_DIR / label.lower() / (stem + ext)
        count += 1

        dst.parent.mkdir(parents=True, exist_ok=True)
        # copy2(src, dst)
        symlink(src, dst)
    break

print(f"Built {count} files.")
