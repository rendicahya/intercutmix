import json
import multiprocessing
import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path, PosixPath
from typing import Union

import cv2
from assertpy.assertpy import assert_that
from python_config import Config
from python_file import count_dir
from python_image import load_image_dir
from python_video import frames_to_video, video_frames, video_info
from tqdm import tqdm


def actorcutmix(
    actor_path: Union[Path, str],
    scene_path: Union[Path, str],
    mask_path: Union[Path, str],
):
    assert_that(actor_path).is_file().is_readable()
    assert_that(mask_path).is_directory().is_readable()

    actor_frames = video_frames(actor_path, reader="decord")
    mask_frames = load_image_dir(mask_path, flag=cv2.IMREAD_GRAYSCALE)
    scene_frame = None

    for actor_frame in actor_frames:
        if scene_frame is None:
            scene_frames = video_frames(scene_path, reader="decord")
            scene_frame = next(scene_frames)

        actor_mask = next(mask_frames)
        scene_mask = 255 - actor_mask

        actor = cv2.bitwise_and(actor_frame, actor_frame, mask=actor_mask)
        scene = cv2.bitwise_and(scene_frame, scene_frame, mask=scene_mask)

        mix = actor + scene
        scene_frame = next(scene_frames, None)

        yield mix


def actorcutmix_job(
    file: PosixPath,
    scene_path: PosixPath,
    mask_path: PosixPath,
    output_path: PosixPath,
    fps: float,
    bar,
):
    bar.set_description(file.stem)

    output_frames = actorcutmix(file, scene_path, mask_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_to_video(
        output_frames,
        output_path,
        writer="moviepy",
        fps=fps,
        codec="mp4v",
    )

    bar.update(1)


print("Performing checks...")

conf = Config("config.json")
assert_that("config.json").is_file().is_readable()

dataset_dir = Path(conf.mix.dataset.path)
scene_dir = Path(conf.mix.scene.path)
mask_dir = Path(conf.mix.mask)
output_dir = Path(conf.mix.output)

assert_that(dataset_dir).is_directory().is_readable()
assert_that(scene_dir).is_directory().is_readable()
assert_that(conf.mix.scene.list).is_file().is_readable()

n_videos = count_files(dataset_dir, ext=conf.mix.dataset.ext)
n_scene_actions = count_dir(scene_dir)

assert (
    n_videos == conf.mix.dataset.n_videos
), f"{conf.mix.dataset.n_videos} videos expected but {n_videos} found."

assert type(conf.mix.n_mix_per_video) == int

with open(conf.mix.scene.list) as f:
    scene_json = json.load(f)

for action, files in scene_json.items():
    for file in files:
        assert_that(scene_dir / file).is_file().is_readable()

print("All checks passed.")

n_cores = multiprocessing.cpu_count()
print(f"Running jobs on {n_cores} cores...")

bar = tqdm(total=n_videos * n_scene_actions * conf.mix.n_mix_per_video)

with ThreadPoolExecutor(max_workers=n_cores) as executor:
    futures = []

    for action in dataset_dir.iterdir():
        for file in action.iterdir():
            mask_path = mask_dir / action.name / file.stem
            fps = video_info(file)["fps"]

            for scene_action in scene_json.keys():
                if action.name == scene_action:
                    continue

                for i in range(conf.mix.n_mix_per_video):
                    scene_list = scene_json[scene_action]
                    scene_pick = random.choice(scene_list)
                    scene_path = scene_dir / scene_pick
                    output_path = (
                        output_dir / action.name / f"{file.stem}-{scene_action}-{i+1}"
                    ).with_suffix(".mp4")

                    if output_path.exists():
                        continue

                    if conf.mix.multithread:
                        futures.append(
                            executor.submit(
                                partial(
                                    actorcutmix_job,
                                    file,
                                    scene_path,
                                    mask_path,
                                    output_path,
                                    fps,
                                    bar,
                                )
                            )
                        )
                    else:
                        actorcutmix_job(
                            file, scene_path, mask_path, output_path, fps, bar
                        )

bar.close()
