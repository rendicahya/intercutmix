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

mid_dir = ROOT / "data" / DATASET / DETECTOR / DET_CONFIDENCE / AUG_METHOD
mix_dir = "mix" if SEED is None else f"mix-{SEED}"

if AUG_METHOD in ("allcutmix", "actorcutmix"):
    mask_in_dir = mid_dir / ("REPP/mask" if USE_REPP else "mask")
    video_out_dir = mid_dir / (f"REPP/{mix_dir}" if USE_REPP else mix_dir)
else:
    relevancy_method = conf.active.relevancy.method
    relevancy_thresh = str(conf.active.relevancy.threshold)

    mask_in_dir = (
        mid_dir
        / ("REPP/mask" if USE_REPP else "mask")
        / relevancy_method
        / relevancy_thresh
    )
    video_out_dir = (
        mid_dir
        / (f"REPP/{mix_dir}" if USE_REPP else mix_dir)
        / relevancy_method
        / relevancy_thresh
    )

print("n videos:", N_VIDEOS)
print("MULTIPLICATION:", MULTIPLICATION)
print("Mask:", mask_in_dir.relative_to(ROOT))
print("Scene:", SCENE_DIR.relative_to(ROOT))
print("Output:", video_out_dir.relative_to(ROOT))

assert_that(VIDEO_IN_DIR).is_directory().is_readable()
assert_that(mask_in_dir).is_directory().is_readable()
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
n_skipped = 0
n_written = 0

for file in VIDEO_IN_DIR.glob(f"**/*{IN_EXT}"):
    action = file.parent.name
    output_action_dir = video_out_dir / action
    mask_path = mask_in_dir / action / file.with_suffix(".npz").name

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
            video_out_dir / action / f"{file.stem}-{scene_class_pick}"
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
print("Skipped videos:", n_skipped)
