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

        yield (
            cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if file_path.exists()
            else None
        )


def cutmix(actor_path, scene_path, mask_path, video_reader):
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
    mask_bundle = np.load(mask_path)["arr_0"]
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
    conf = Config("config.json")
    video_in_dir = Path(conf[conf.active.dataset].path)
    scene_dir = Path(conf.cutmix.input[conf.active.dataset].scene.path)

    dataset = conf.active.dataset
    detector = conf.active.detector
    mode = conf.active.mode
    relevancy_model = conf.relevancy.active.method
    relevancy_threshold = conf.relevancy.active.threshold

    print("Dataset:", dataset)
    print("Mode:", mode)
    print("REPP:", conf.cutmix.use_REPP)

    if conf.cutmix.use_REPP:
        mask_dir = (
            Path("data")
            / dataset
            / "REPP"
            / mode
            / "mask"
            / relevancy_model
            / str(relevancy_threshold)
        )
        video_out_dir = (
            Path("data")
            / dataset
            / "REPP"
            / mode
            / "mix"
            / relevancy_model
            / str(relevancy_threshold)
        )
    else:
        mask_dir = (
            Path("data")
            / dataset
            / detector
            / "select"
            / mode
            / "mask"
            / relevancy_model
            / str(relevancy_threshold)
        )
        video_out_dir = (
            Path("data")
            / dataset
            / detector
            / "select"
            / mode
            / "mix"
            / relevancy_model
            / str(relevancy_threshold)
        )

    out_ext = conf.cutmix.output.ext
    scene_options = conf.cutmix.input[conf.active.dataset].scene.list
    n_videos = count_files(video_in_dir, ext=conf[conf.active.dataset].ext)
    n_scene_actions = count_dir(scene_dir)

    print("Performing checks...")

    assert_that(video_in_dir).is_directory().is_readable()
    assert_that(scene_dir).is_directory().is_readable()
    assert_that(scene_options).is_file().is_readable()

    with open(scene_options) as f:
        scene_json = json.load(f)

    for action, files in scene_json.items():
        for file in files:
            assert_that(scene_dir / file).is_file().is_readable()

    print("All checks passed.")

    random.seed(conf.random_seed)

    bar = tqdm(total=n_videos * conf.cutmix.multiplication)
    n_error = 0

    for action in video_in_dir.iterdir():
        output_action_dir = video_out_dir / action.name
        n_target_videos = count_files(action) * conf.cutmix.multiplication

        if output_action_dir.exists():
            if count_files(output_action_dir) == n_target_videos:
                print(f'Action "{action.name}" is complete. Skipping...')
                bar.update(n_target_videos)

                continue
            else:
                print(
                    f'Action "{action.name}" is partially complete. Deleting and remixing...'
                )
                shutil.rmtree(output_action_dir)

        for file in action.iterdir():
            mask_path = mask_dir / action.name / file.with_suffix(".npz").name
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

                bar.set_description(file.stem[:50].ljust(50))
                output_path.parent.mkdir(parents=True, exist_ok=True)

                out_frames = cutmix(
                    file, scene_path, mask_path, conf.active.video.reader
                )

                if not out_frames:
                    n_error += 1
                    continue

                frames_to_video(
                    out_frames,
                    output_path,
                    writer=conf.active.video.writer,
                    fps=fps,
                )

                bar.update(1)

    bar.close()
    print("Errors:", n_error)
