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

root = Path.cwd()
dataset = conf.active.dataset
detector = conf.active.detector
object_conf = str(conf.unidet.select.confidence)
method = conf.active.mode
use_REPP = conf.active.use_REPP
video_in_dir = root / conf[dataset].path
scene_dir = root / conf[dataset].scene.path
scene_options = scene_dir / "list.txt"
multiplication = conf.cutmix.multiplication
video_ext = conf[dataset].ext
n_videos = conf[dataset].n_videos
random_seed = conf.active.random_seed
out_ext = conf.cutmix.output_ext

mid_dir = root / "data" / dataset / detector / object_conf / method
mix_dir = "mix" if random_seed is None else f"mix-{random_seed}"

if method in ("allcutmix", "actorcutmix"):
    mask_in_dir = mid_dir / ("REPP/mask" if use_REPP else "mask")
    video_out_dir = mid_dir / (f"REPP/{mix_dir}" if use_REPP else mix_dir)
else:
    relevancy_method = conf.active.relevancy.method
    relevancy_thresh = str(conf.active.relevancy.threshold)

    mask_in_dir = (
        mid_dir
        / ("REPP/mask" if use_REPP else "mask")
        / relevancy_method
        / relevancy_thresh
    )
    video_out_dir = (
        mid_dir
        / (f"REPP/{mix_dir}" if use_REPP else mix_dir)
        / relevancy_method
        / relevancy_thresh
    )

print("n videos:", n_videos)
print("Multiplication:", multiplication)
print("Random seed:", random_seed)
print("Mask:", mask_in_dir.relative_to(root))
print("Scene:", scene_dir.relative_to(root))
print("Output:", video_out_dir.relative_to(root))

assert_that(video_in_dir).is_directory().is_readable()
assert_that(mask_in_dir).is_directory().is_readable()
assert_that(scene_dir).is_directory().is_readable()
assert_that(scene_options).is_file().is_readable()

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

print("Checking files...")

scene_dict = defaultdict(list)

with open(scene_options) as file:
    for line in file:
        action, filename = line.split()[0].split("/")

        scene_dict[action].append(filename)
        assert_that(scene_dir / action / filename).is_file().is_readable()

random.seed(random_seed)

bar = tqdm(total=n_videos * multiplication, dynamic_ncols=True)
n_skipped = 0
n_written = 0

for file in video_in_dir.glob(f"**/*{video_ext}"):
    action = file.parent.name
    output_action_dir = video_out_dir / action
    mask_path = mask_in_dir / action / file.with_suffix(".npz").name

    if not mask_path.is_file() or not mask_path.exists():
        continue

    mask_bundle = np.load(mask_path)["arr_0"]
    fps = mmcv.VideoReader(str(file)).fps
    scene_class_options = [s for s in scene_dict.keys() if s != action]

    for i in range(multiplication):
        bar.set_description(f"{file.stem[:40]} ({i+1}/{multiplication})")

        scene_class_pick = random.choice(scene_class_options)
        scene_options = scene_dict[scene_class_pick]
        scene_pick = random.choice(scene_options)
        scene_path = scene_dir / scene_class_pick / scene_pick
        output_path = (
            video_out_dir / action / f"{file.stem}-{scene_class_pick}"
        ).with_suffix(out_ext)

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
