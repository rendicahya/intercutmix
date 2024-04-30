import sys

sys.path.append(".")

import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import cv2
import numpy as np
from assertpy.assertpy import assert_that
from config import settings as conf
from python_file import count_dir, count_files
from python_video import frames_to_video, video_frames, video_info
from tqdm import tqdm


def thread_job(
    file, scene_path, mask_bundle, video_reader, video_writer, fps, output_path, bar
):
    out_frames = cutmix(file, scene_path, mask_bundle, video_reader)

    if out_frames:
        frames_to_video(
            out_frames,
            output_path,
            writer=video_writer,
            fps=fps,
        )
    else:
        print("out_frames None: ", file.name)

    bar.update(1)


def cutmix(actor_path, scene_path, mask_bundle, video_reader):
    if not actor_path.is_file() or not actor_path.exists():
        print("Not a file or not exists:", actor_path)
        return None

    if not scene_path.is_file() or not scene_path.exists():
        print("Not a file or not exists:", scene_path)
        return None

    if not mask_path.is_file() or not mask_path.exists():
        print("Not a file or not exists:", mask_path)
        return None

    actor_frames = video_frames(actor_path, reader=video_reader)
    info = video_info(actor_path)
    w, h = info["width"], info["height"]
    blank = np.zeros((h, w), np.uint8)
    scene_frame = None
    scene_info = video_info(scene_path)
    scene_w, scene_h = scene_info["width"], scene_info["height"]

    for f, actor_frame in enumerate(actor_frames):
        if f == len(mask_bundle) - 1:
            return

        if scene_frame is None:
            scene_frames = video_frames(scene_path, reader=video_reader)
            scene_frame = next(scene_frames)

        if scene_w != w or scene_h != h:
            scene_frame = cv2.resize(scene_frame, (w, h))

        actor_mask = mask_bundle[f]

        if actor_mask is None:
            actor_mask = blank

        scene_mask = 255 - actor_mask

        actor = cv2.bitwise_and(actor_frame, actor_frame, mask=actor_mask)
        scene = cv2.bitwise_and(scene_frame, scene_frame, mask=scene_mask)

        mix = actor + scene
        scene_frame = next(scene_frames, None)

        yield mix


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
    n_videos = count_files(video_in_dir, ext=conf[dataset].ext)
    random_seed = conf.active.random_seed
    n_threads = conf.cutmix.n_threads

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

    print("Dataset:", dataset)
    print("Detector:", detector)
    print("Object selection:", object_selection)
    print("Mode:", mode)
    print("REPP:", use_REPP)
    print("Relevancy model:", relevancy_model)
    print("Relevancy threshold:", relevancy_thresh)
    print("Σ videos:", n_videos)
    print("Multiplication:", multiplication)
    print("Use smooth mask:", use_smooth_mask)
    print("Seed:", random_seed)
    print("Threads:", n_threads)
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
    executor = ThreadPoolExecutor(max_workers=n_threads)
    futures = []

    for action in video_in_dir.iterdir():
        output_action_dir = video_out_dir / action.name

        for file in action.iterdir():
            mask_path = mask_in_dir / action.name / file.with_suffix(".npz").name

            if not mask_path.is_file() or not mask_path.exists():
                continue

            mask_bundle = np.load(mask_path)["arr_0"]
            fps = video_info(file)["fps"]
            scene_class_options = [s for s in scene_dict.keys() if s != action.name]

            for i in range(multiplication):
                scene_class_pick = random.choice(scene_class_options)
                scene_options = scene_dict[scene_class_pick]
                scene_pick = random.choice(scene_options)
                scene_path = scene_dir / scene_class_pick / scene_pick
                output_path = (
                    video_out_dir / action.name / f"{file.stem}-{scene_class_pick}"
                ).with_suffix(out_ext)

                scene_class_options.remove(scene_class_pick)

                if output_path.exists() and video_info(output_path)["n_frames"] > 0:
                    continue

                output_path.parent.mkdir(parents=True, exist_ok=True)

                future = executor.submit(
                    thread_job,
                    file,
                    scene_path,
                    mask_bundle,
                    conf.active.video.reader,
                    conf.active.video.writer,
                    fps,
                    output_path,
                    bar,
                )

                futures.append(future)

    for future in futures:
        future.result()

    executor.shutdown()

    bar.close()
