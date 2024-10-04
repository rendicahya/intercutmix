import sys

sys.path.append(".")

import json
from pathlib import Path
from shutil import copy2

from tqdm import tqdm

from config import settings as conf

root = Path.cwd()
datasets = "ucf101", "hmdb51", "kinetics100"
uhk_interaction_dir = root / "data/uhk_interaction/videos"
detector = conf.active.detector
use_REPP = conf.active.use_REPP
det_confidence = str(conf.unidet.select.confidence)
uhk_interaction_dir.mkdir(parents=True, exist_ok=True)

for dataset in datasets:
    dataset_dir = Path(conf[dataset].path)
    ext = conf[dataset].ext
    n_videos = conf[dataset].n_videos
    action_index = {}
    relevancy_method = conf.active.relevancy.method
    relevancy_thresh = str(conf.active.relevancy.threshold)
    count = 0
    acm_mask_dir = (
        root
        / "data"
        / dataset
        / detector
        / det_confidence
        / "actorcutmix"
        / ("REPP/mask" if use_REPP else "mask")
    )
    icm_mask_dir = (
        root
        / "data"
        / dataset
        / detector
        / det_confidence
        / "intercutmix"
        / ("REPP/mask" if use_REPP else "mask")
        / relevancy_method
        / relevancy_thresh
    )

    with open(acm_mask_dir / "ratio.json") as f:
        acm_ratio_json = json.load(f)

    with open(icm_mask_dir / "ratio.json") as f:
        icm_ratio_json = json.load(f)

    for file in dataset_dir.glob(f"**/*{ext}"):
        action = file.parent.name

        action_index.update({file.stem: action})

    print(f"Building from {dataset}...")

    for file in tqdm(acm_ratio_json.keys(), dynamic_ncols=True):
        acm_ratio = acm_ratio_json[file]
        icm_ratio = icm_ratio_json[file]

        if icm_ratio <= acm_ratio:
            continue

        action = action_index[file]
        src = dataset_dir / action / (file + ext)
        dst = uhk_interaction_dir / action / (file + ext)
        count += 1

        dst.parent.mkdir(parents=True, exist_ok=True)
        copy2(src, dst)

    print(f"Built {count} files.")
