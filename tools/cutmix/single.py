import sys

sys.path.append(".")

import random
from collections import defaultdict
from pathlib import Path

import click
import mmcv
import numpy as np
from cutmix import cutmix
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf
from python_video import frames_to_video

ROOT = Path.cwd()
DATASET = conf.active.DATASET
DETECTOR = conf.active.DETECTOR
DET_CONFIDENCE = str(conf.unidet.select.confidence)
AUG_METHOD = conf.active.mode
USE_REPP = conf.active.USE_REPP
VIDEO_IN_DIR = ROOT / conf[DATASET].path
SCENE_DIR = ROOT / conf[DATASET].scene.path
scene_options = SCENE_DIR / "list.txt"
MULTIPLICATION = conf.cutmix.MULTIPLICATION
IN_EXT = conf[DATASET].ext
N_VIDEOS = conf[DATASET].N_VIDEOS
SEED = conf.active.random_seed
OUT_EXT = conf.cutmix.output_ext

MID_DIR = ROOT / "data" / DATASET / DETECTOR / DET_CONFIDENCE / AUG_METHOD
MIX_DIR = "mix" if SEED is None else f"mix-{SEED}"

if AUG_METHOD in ("allcutmix", "actorcutmix"):
    MASK_IN_DIR = MID_DIR / ("REPP/mask" if USE_REPP else "mask")
    VIDEO_OUT_DIR = MID_DIR / (f"REPP/{MIX_DIR}" if USE_REPP else MIX_DIR)
else:
    RELEV_METHOD = conf.active.relevancy.method
    RELEV_THRESH = str(conf.active.relevancy.threshold)

    MASK_IN_DIR = (
        MID_DIR / ("REPP/mask" if USE_REPP else "mask") / RELEV_METHOD / RELEV_THRESH
    )
    VIDEO_OUT_DIR = (
        MID_DIR
        / (f"REPP/{MIX_DIR}" if USE_REPP else MIX_DIR)
        / RELEV_METHOD
        / RELEV_THRESH
    )

print("n videos:", N_VIDEOS)
print("Multiplication:", MULTIPLICATION)
print("Mask:", MASK_IN_DIR.relative_to(ROOT))
print("Scene:", SCENE_DIR.relative_to(ROOT))
print(
    "Output:",
    VIDEO_OUT_DIR.relative_to(ROOT),
    "(exists)" if VIDEO_OUT_DIR.exists() else "(not exists)",
)

assert_that(VIDEO_IN_DIR).is_directory().is_readable()
assert_that(MASK_IN_DIR).is_directory().is_readable()
assert_that(SCENE_DIR).is_directory().is_readable()
assert_that(scene_options).is_file().is_readable()

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

print("Checking files...")

scene_dict = defaultdict(list)

with open(scene_options) as file:
    for line in file:
        action, filename = line.split()[0].split("/")

        scene_dict[action].append(filename)
        assert_that(SCENE_DIR / action / filename).is_file().is_readable()

random.seed(SEED)

bar = tqdm(total=N_VIDEOS * MULTIPLICATION, dynamic_ncols=True)
n_written = 0

for file in VIDEO_IN_DIR.glob(f"**/*{IN_EXT}"):
    action = file.parent.name
    output_action_dir = VIDEO_OUT_DIR / action
    mask_path = MASK_IN_DIR / action / file.with_suffix(".npz").name

    if not mask_path.is_file() or not mask_path.exists():
        continue

    mask_bundle = np.load(mask_path)["arr_0"]
    fps = mmcv.VideoReader(str(file)).fps
    scene_class_options = [s for s in scene_dict.keys() if s != action]

    for i in range(MULTIPLICATION):
        bar.set_description(f"{file.stem[:40]} ({i+1}/{MULTIPLICATION})")

        scene_class_pick = random.choice(scene_class_options)
        scene_options = scene_dict[scene_class_pick]
        scene_pick = random.choice(scene_options)
        scene_path = SCENE_DIR / scene_class_pick / scene_pick
        output_path = (
            VIDEO_OUT_DIR / action / f"{file.stem}-{scene_class_pick}"
        ).with_suffix(OUT_EXT)

        scene_class_options.remove(scene_class_pick)

        if output_path.exists() and mmcv.VideoReader(str(output_path)).frame_cnt > 0:
            bar.update(1)
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)

        out_frames = cutmix(file, scene_path, mask_bundle)

        if out_frames:
            frames_to_video(
                out_frames,
                output_path,
                writer=conf.active.video.writer,
                fps=fps,
            )

            n_written += 1
        else:
            print("out_frames None: ", file.name)

        bar.update(1)

bar.close()
print("Written videos:", n_written)
