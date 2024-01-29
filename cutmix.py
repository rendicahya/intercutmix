import json
import random
import shutil
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
        file_path = dir_path / f"{i:05}.png"

        yield cv2.imread(
            str(file_path), cv2.IMREAD_GRAYSCALE
        ) if file_path.exists() else None


def cutmix(actor_path, scene_path, mask_path, video_reader):
    assert_that(actor_path).is_file().is_readable()
    assert_that(mask_path).is_directory().is_readable()

    actor_frames = video_frames(actor_path, reader=video_reader)
    info = video_info(actor_path)
    w, h = info["width"], info["height"]
    blank = np.zeros((h, w), np.uint8)
    mask_frames = load_image_dir(mask_path, info["n_frames"])
    scene_frame = None

    for actor_frame in actor_frames:
        if scene_frame is None:
            scene_frames = video_frames(scene_path, reader=video_reader)
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


if __name__ == "__main__":
    print("Performing checks...")

    conf = Config("config.json")
    assert_that("config.json").is_file().is_readable()

    video_in_dir = Path(conf.cutmix.video.path)
    scene_dir = Path(conf.cutmix.scene.path)
    mask_dir = Path(conf.cutmix.mask.path)
    video_out_dir = Path(conf.cutmix.output.path)
    out_ext = conf.cutmix.output.ext
    scene_options = conf.cutmix.scene.list
    n_videos = count_files(video_in_dir, ext=conf.cutmix.video.ext)
    n_scene_actions = count_dir(scene_dir)

    assert_that(video_in_dir).is_directory().is_readable()
    assert_that(scene_dir).is_directory().is_readable()
    assert_that(scene_options).is_file().is_readable()

    with open(scene_options) as f:
        scene_json = json.load(f)

    for action, files in scene_json.items():
        for file in files:
            assert_that(scene_dir / file).is_file().is_readable()

    print("All checks passed.")

    n_video_blacklist = len(conf.cutmix.video.blacklist)
    n_target_videos = (n_videos - n_video_blacklist) * conf.cutmix.multiplication
    action_whitelist = conf.cutmix.action.whitelist
    action_blacklist = conf.cutmix.action.blacklist
    bar = tqdm(total=n_target_videos)

    for action in video_in_dir.iterdir():
        if (action_whitelist is not None and action.name not in action_whitelist) or (
            action_blacklist is not None and action.name in action_blacklist
        ):
            continue

        output_action_dir = video_out_dir / action.name

        n_video_blacklist = sum(
            1 for v in conf.cutmix.video.blacklist if v.split("_")[1] == action.name
        )

        n_target_videos = (
            count_files(action) - n_video_blacklist
        ) * conf.cutmix.multiplication

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
            if file.stem in conf.cutmix.video.blacklist:
                print(f"{file.name} skipped")
                bar.update(1)

                continue

            mask_path = mask_dir / action.name / file.stem
            fps = video_info(file)["fps"]
            scene_class_options = [s for s in scene_json.keys() if s != action.name]

            for i in range(conf.cutmix.multiplication):
                scene_class_pick = random.choice(scene_class_options)
                scene_options = scene_json[scene_class_pick]
                scene_pick = random.choice(scene_options)
                scene_path = scene_dir / scene_pick
                output_path = (
                    video_out_dir / action.name / f"{file.stem}-{scene_class_pick}"
                ).with_suffix(out_ext)

                scene_class_options.remove(scene_class_pick)

                if output_path.exists():
                    bar.set_description("Skipping finished videos...")
                    bar.update(1)
                    continue

                bar.set_description(file.stem)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                output_frames = cutmix(
                    file, scene_path, mask_path, conf.cutmix.video.reader
                )

                frames_to_video(
                    output_frames,
                    output_path,
                    writer=conf.cutmix.video.writer,
                    fps=fps,
                )

                bar.update(1)

    bar.close()
