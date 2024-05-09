import sys

sys.path.append(".")

import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import click
import cv2
import mmcv
import numpy as np
from assertpy.assertpy import assert_that
from config import settings as conf
from cutmix import cutmix
from python_file import count_dir, count_files
from python_video import frames_to_video
from tqdm import tqdm


if __name__ == "__main__":
    dataset = conf.active.dataset
    detector = conf.active.detector
    object_selection = conf.active.object_selection
    mode = conf.active.mode
    use_REPP = conf.active.use_REPP
    relevancy_model = conf.relevancy.active.method
    relevancy_thresh = str(conf.relevancy.active.threshold)
    video_in_dir = Path(conf[dataset].path)
    scene_dir = Path(conf[dataset].scene.path)
    scene_options = scene_dir / "list.txt"
    multiplication = conf.cutmix.multiplication
    use_smooth_mask = conf.active.smooth_mask.enabled
    video_ext = conf[dataset].ext
    n_videos = count_files(video_in_dir, ext=video_ext)
    random_seed = conf.active.random_seed

    method = "select" if object_selection else "detect"
    method_dir = Path("data") / dataset / detector / method
    mix_mode = "mix" if random_seed is None else f"mix-{random_seed}"

    if method == "detect":
        mask_in_dir = method_dir / ("REPP/mask" if use_REPP else "mask")
        video_out_dir = method_dir / (f"REPP/{mix_mode}" if use_REPP else mix_mode)
    elif method == "select":
        mask_in_dir = method_dir / mode / ("REPP/mask" if use_REPP else "mask")
        video_out_dir = (
            method_dir / mode / (f"REPP/{mix_mode}" if use_REPP else mix_mode)
        )

        if mode == "intercutmix":
            mask_in_dir = mask_in_dir / relevancy_model / relevancy_thresh
            video_out_dir = video_out_dir / relevancy_model / relevancy_thresh

    print("Object selection:", object_selection)
    print("Mode:", mode)
    print("REPP:", use_REPP)
    print("Relevancy model:", relevancy_model)
    print("Relevancy threshold:", relevancy_thresh)
    print("Σ videos:", n_videos)
    print("Multiplication:", multiplication)
    print("Use smooth mask:", use_smooth_mask)
    print("Seed:", random_seed)
    print("Input:", mask_in_dir)
    print("Output:", video_out_dir)

    out_ext = conf.cutmix.output.ext
    n_scene_actions = count_dir(scene_dir)

    assert_that(video_in_dir).is_directory().is_readable()
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

    if random_seed is not None:
        random.seed(random_seed)

    bar = tqdm(total=n_videos * multiplication)
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
            scene_class_pick = random.choice(scene_class_options)
            scene_options = scene_dict[scene_class_pick]
            scene_pick = random.choice(scene_options)
            scene_path = scene_dir / scene_class_pick / scene_pick
            output_path = (
                video_out_dir / action / f"{file.stem}-{scene_class_pick}"
            ).with_suffix(out_ext)

            scene_class_options.remove(scene_class_pick)

            if (
                output_path.exists()
                and mmcv.VideoReader(str(output_path)).frame_cnt > 0
            ):
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
