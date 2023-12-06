import json
import multiprocessing
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path, PosixPath
from typing import Union

import cv2
from assertpy.assertpy import assert_that
from python_config import Config
from python_file import count_dir, count_files
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

    actor_frames = video_frames(actor_path, reader=conf.mix.video.reader)
    mask_frames = load_image_dir(mask_path, flag=cv2.IMREAD_GRAYSCALE)
    scene_frame = None

    for actor_frame in actor_frames:
        if scene_frame is None:
            scene_frames = video_frames(scene_path, reader=conf.mix.video.reader)
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
):
    bar.set_description(file.stem)

    output_frames = actorcutmix(file, scene_path, mask_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_to_video(
        output_frames,
        output_path,
        writer=conf.mix.video.writer,
        fps=fps,
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

with open(conf.mix.scene.list) as f:
    scene_json = json.load(f)

for action, files in scene_json.items():
    for file in files:
        assert_that(scene_dir / file).is_file().is_readable()

print("All checks passed.")

n_cores = multiprocessing.cpu_count()
n_skip_videos = len(conf.mix.skip_videos)
bar = tqdm(total=(n_videos - n_skip_videos) * conf.mix.multiplication)

if conf.mix.multithread:
    print(f"Running on {n_cores} cores...")

with ThreadPoolExecutor(max_workers=n_cores) as executor:
    futures = []

    for action in dataset_dir.iterdir():
        output_action_dir = output_dir / action.name

        n_skip_videos = sum(
            1 for v in conf.mix.skip_videos if v.split("_")[1] == action.name
        )

        n_target_videos = (
            count_files(action) - n_skip_videos
        ) * conf.mix.multiplication

        if output_action_dir.exists():
            if count_files(output_action_dir) == n_target_videos:
                print(f"Action {action.name} is complete. Skipping...")
                bar.update(n_target_videos)

                continue
            else:
                print(
                    f"Action {action.name} is partially complete. Deleting and remixing..."
                )
                shutil.rmtree(output_action_dir)

        for file in action.iterdir():
            if file.stem in conf.mix.skip_videos:
                print(f"{file.name} skipped")
                bar.update(1)

                continue

            mask_path = mask_dir / action.name / file.stem
            fps = video_info(file)["fps"]
            scene_class_option = [s for s in scene_json.keys() if s != action.name]

            for i in range(conf.mix.multiplication):
                scene_class_pick = random.choice(scene_class_option)
                scene_list = scene_json[scene_class_pick]
                scene_pick = random.choice(scene_list)
                scene_path = scene_dir / scene_pick
                output_path = (
                    output_dir / action.name / f"{file.stem}-{scene_class_pick}"
                ).with_suffix(".mp4")

                scene_class_option.remove(scene_class_pick)

                if output_path.exists():
                    bar.set_description("Skipping finished videos...")
                    bar.update(1)
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
                            )
                        )
                    )
                else:
                    actorcutmix_job(
                        file,
                        scene_path,
                        mask_path,
                        output_path,
                        fps,
                    )

bar.close()
