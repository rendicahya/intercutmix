import json
import multiprocessing
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from assertpy.assertpy import assert_that
from python_config import Config
from python_file import count_dir, count_files
from python_video import frames_to_video, video_frames, video_info
from tqdm import tqdm


def load_image_dir(dir_path, n_frames):
    assert_that(dir_path).is_directory().is_readable()

    for i in range(n_frames):
        file_path = dir_path / f"{i:05}{conf.mix.mask.ext}"

        yield cv2.imread(
            str(file_path), cv2.IMREAD_GRAYSCALE
        ) if file_path.exists() else None


def actorcutmix(
    actor_path,
    scene_path,
    mask_path,
):
    assert_that(actor_path).is_file().is_readable()
    assert_that(mask_path).is_directory().is_readable()

    actor_frames = video_frames(actor_path, reader=conf.mix.video.reader)
    info = video_info(actor_path)
    w, h = info["width"], info["height"]
    blank = np.zeros((h, w), np.uint8)
    mask_frames = load_image_dir(mask_path, info["n_frames"])
    scene_frame = None

    for actor_frame in actor_frames:
        if scene_frame is None:
            scene_frames = video_frames(scene_path, reader=conf.mix.video.reader)
            scene_frame = next(scene_frames)

        actor_mask = next(mask_frames)

        if actor_mask is None:
            actor_mask = blank

        scene_mask = 255 - actor_mask

        actor = cv2.bitwise_and(actor_frame, actor_frame, mask=actor_mask)
        scene = cv2.bitwise_and(scene_frame, scene_frame, mask=scene_mask)

        mix = actor + scene
        scene_frame = next(scene_frames, None)

        yield mix


def actorcutmix_job(
    file,
    scene_path,
    mask_path,
    output_path,
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

dataset_root = Path(conf.mix.dataset.path)
scene_root = Path(conf.mix.scene.path)
mask_root = Path(conf.mix.mask.path)
output_root = Path(conf.mix.output.path)
output_ext = conf.mix.output.ext

assert_that(dataset_root).is_directory().is_readable()
assert_that(scene_root).is_directory().is_readable()
assert_that(conf.mix.scene.list).is_file().is_readable()

n_videos = count_files(dataset_root, ext=conf.mix.dataset.ext)
n_scene_actions = count_dir(scene_root)

assert (
    n_videos == conf.mix.dataset.n_videos
), f"{conf.mix.dataset.n_videos} videos expected but {n_videos} found."

with open(conf.mix.scene.list) as f:
    scene_json = json.load(f)

for action, files in scene_json.items():
    for file in files:
        assert_that(scene_root / file).is_file().is_readable()

print("All checks passed.")

n_cores = multiprocessing.cpu_count()
n_video_blacklist = len(conf.mix.video.blacklist)
n_target_videos = (n_videos - n_video_blacklist) * conf.mix.multiplication
action_whitelist = conf.mix.action.whitelist
action_blacklist = conf.mix.action.blacklist
bar = tqdm(total=n_target_videos)

if conf.mix.multithread:
    print(f"Running on {n_cores} cores...")

with ThreadPoolExecutor(max_workers=n_cores) as executor:
    futures = []

    for action in dataset_root.iterdir():
        if (action_whitelist is not None and action.name not in action_whitelist) or (
            action_blacklist is not None and action.name in action_blacklist
        ):
            continue

        output_action_dir = output_root / action.name

        n_video_blacklist = sum(
            1 for v in conf.mix.video.blacklist if v.split("_")[1] == action.name
        )

        n_target_videos = (
            count_files(action) - n_video_blacklist
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
            if file.stem in conf.mix.video.blacklist:
                print(f"{file.name} skipped")
                bar.update(1)

                continue

            mask_path = mask_root / action.name / file.stem
            fps = video_info(file)["fps"]
            scene_class_option = [s for s in scene_json.keys() if s != action.name]

            for i in range(conf.mix.multiplication):
                scene_class_pick = random.choice(scene_class_option)
                scene_list = scene_json[scene_class_pick]
                scene_pick = random.choice(scene_list)
                scene_path = scene_root / scene_pick
                output_path = (
                    output_root / action.name / f"{file.stem}-{scene_class_pick}"
                ).with_suffix(output_ext)

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
