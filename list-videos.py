import json
from pathlib import Path

import numpy as np
from assertpy.assertpy import assert_that
from python_config import Config
from tqdm import tqdm

conf = Config("config.json")
relevancy_model = conf.relevancy.active.method
relevancy_threshold = conf.relevancy.active.threshold
dataset = conf.active.dataset
detector = conf.active.detector
mode = conf.active.mode
ext = conf.cutmix.output.ext
multiplication = conf.cutmix.multiplication
n_videos = conf[conf.active.dataset].n_videos * multiplication
n_actions = conf[conf.active.dataset].n_classes
use_REPP = conf.cutmix.use_REPP
video_root = Path(conf[dataset].path).parent

if use_REPP:
    mode_dir = video_root / "REPP" / mode
else:
    mode_dir = video_root / detector / "select" / mode

video_dir = mode_dir / "mix" / relevancy_model / str(relevancy_threshold)

print("Dataset:", dataset)
print("Mode:", mode)
print("REPP:", use_REPP)
print("Relevancy model:", relevancy_model)
print("Relevancy thresh.:", relevancy_threshold)
print("N videos:", n_videos)
print("Input:", video_dir)
print("Output:", video_dir / "list.json")

assert_that(video_dir).is_directory().is_readable()

data = {}
bar = tqdm(total=n_actions)

for action in sorted(video_dir.iterdir()):
    files = [
        str(file.relative_to(video_dir))
        for file in action.iterdir()
        if file.is_file() and file.suffix == ext
    ]

    data[action.name] = files
    bar.update(1)

bar.close()

with open(video_dir / "list.json", "w") as f:
    json.dump(data, f, indent=2)
