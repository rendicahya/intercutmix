import sys

sys.path.append(".")

import json
import re
from collections import defaultdict
from os import symlink
from pathlib import Path

from config import settings as conf

ROOT = Path.cwd()
DETECTOR = conf.active.detector
OUT_DIR = ROOT / "data/interaction/videos"
RELEVANCY = conf.active.relevancy
THRESHOLD = 0.005

datasets = "ucf101", "hmdb51", "kinetics100"
greater = defaultdict(list)

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

    stem2label = {
        file.stem: file.parent.name
        for file in (dataset_dir / "videos").glob(f"**/*{ext}")
    }

    for stem in acm_json.keys():
        acm_ratio = acm_json[stem]
        icm_ratio = icm_json[stem]

        if icm_ratio <= acm_ratio:
            continue

        label = stem2label[stem]
        path = dataset_dir / "videos" / label / (stem + ext)
        greater[label].append(path)

# camel_to_snake = re.compile(r"(?<=[a-z])([A-Z])")
greater_count = sum(len(videos) for videos in greater.values())
eligible = {
    label: videos
    for label, videos in greater.items()
    if len(videos) >= greater_count * THRESHOLD
}

for label, paths in eligible.items():
    # label = camel_to_snake.sub(r"_\1", label).lower()

    for path in paths:
        dst = OUT_DIR / label / path.name

        dst.parent.mkdir(parents=True, exist_ok=True)
        symlink(path, dst)

eligible_count = sum(len(videos) for videos in eligible.values())
print(eligible_count, "files built.")

# for label in OUT_DIR.iterdir():
#     count = sum(1 for file in label.iterdir() if file.is_file())

#     print(label.name, count)
