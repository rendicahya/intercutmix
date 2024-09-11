import sys

sys.path.append(".")

import pickle
import random
from collections import defaultdict
from pathlib import Path

import click
import mmcv
import numpy as np
from assertpy.assertpy import assert_that
from config import settings as conf
from cutmix import cutmix
from python_file import count_dir, count_files
from python_video import frames_to_video
from tqdm import tqdm


@click.command()
@click.argument(
    "dump-path",
    nargs=1,
    required=True,
    type=click.Path(
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
def main(dump_path):
    dataset = conf.active.dataset
    detector = conf.active.detector
    object_selection = conf.active.object_selection
    mix_mode = conf.active.mode
    use_REPP = conf.active.use_REPP
    relevancy_model = conf.relevancy.active.method
    relevancy_thresh = str(conf.relevancy.active.threshold)
    multiplication = conf.cutmix.mix2train.multiplication
    random_seed = conf.active.random_seed
    action_dir = Path(conf[dataset].path)
    action_list = action_dir / "list.txt"
    work_dir = Path.cwd() / "mmaction2/work_dirs"
    video_out_dir = Path("data") / (
        str(dump_path.parent.parent.relative_to(work_dir)) + "---"
    )
    action_ext = conf[dataset].ext
    scene_ext = conf.cutmix.output.ext
    n_videos = count_files(action_dir, ext=action_ext)

    scene_dir = Path(conf[dataset].scene.path)
    scene_list = scene_dir / "list.txt"

    mix_dir = (
        Path("data")
        / dataset
        / detector
        / "select"
        / mix_mode
        / (f"REPP/{mix_mode}" if use_REPP else mix_mode)
        / relevancy_model
        / relevancy_thresh
    )
    mix_list = mix_dir / "list.txt"

    mask_dir = (
        Path("data")
        / dataset
        / detector
        / "select"
        / mix_mode
        / ("REPP/mask" if use_REPP else "mask")
    )

    if mix_mode == "intercutmix":
        mask_dir = mask_dir / relevancy_model / relevancy_thresh

    mix2train_mode = conf.cutmix.mix2train.test_mode
    file_list_path = scene_list if mix2train_mode == "scene" else mix_list

    with open(dump_path, "rb") as file:
        dump_data = pickle.load(file)

    print("Dataset:", dataset)
    print("Detector:", detector)
    print("Mixing mode:", mix_mode)
    print("REPP:", use_REPP)
    print("Relevancy model:", relevancy_model)
    print("Relevancy threshold:", relevancy_thresh)
    print("Σ videos:", len(dump_data))
    print("Multiplication:", multiplication)
    print("Seed:", random_seed)
    print("Mask dir:", mask_dir)
    print("Dump:", dump_path)
    print("Input:", mask_dir)
    print("Output:", video_out_dir)

    out_ext = conf.cutmix.output.ext
    n_scene_actions = count_dir(scene_dir)

    assert_that(action_dir).is_directory().is_readable()
    assert_that(scene_dir).is_directory().is_readable()
    assert_that(action_list).is_file().is_readable()
    assert_that(scene_list).is_file().is_readable()

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    print("Checking scene videos...")

    action_dict = defaultdict(list)

    with open(scene_list) as file:
        for line in file:
            action, filename = line.split()[0].split("/")

            assert_that(scene_dir / action / filename).is_file().is_readable()

    with open(action_list) as file:
        for line in file:
            action, filename = line.split()[0].split("/")

            action_dict[action].append(filename)

    with open(file_list_path) as file:
        file_list = [line.strip().split()[0].split("/") for line in file]

    random.seed(random_seed)

    bar = tqdm(total=len(dump_data) * multiplication, dynamic_ncols=True)
    n_skipped = 0
    n_written = 0

    for i, dump_item in enumerate(dump_data):
        if dump_item["pred_label"] != dump_item["gt_label"]:
            bar.update(multiplication)
            continue

        scene_action, scene_video = file_list[i]
        scene_path = scene_dir / scene_action / scene_video

        fps = mmcv.VideoReader(str(scene_path)).fps
        action_class_options = [s for s in action_dict.keys() if s != scene_action]

        for _ in range(multiplication):
            action_class_pick = random.choice(action_class_options)
            action_options = action_dict[action_class_pick]
            action_pick = random.choice(action_options)
            action_path = action_dir / action_class_pick / action_pick
            output_path = (
                video_out_dir
                / scene_action
                / f"{scene_video.split('.')[0]}-{action_pick.split('.')[0]}"
            ).with_suffix(out_ext)

            action_class_options.remove(action_class_pick)

            mask_path = (
                mask_dir / action_class_pick / action_pick.replace(action_ext, ".npz")
            )

            if not mask_path.exists() or not mask_path.is_file():
                bar.update(1)
                continue

            mask_bundle = np.load(mask_path)["arr_0"]

            if (
                output_path.exists()
                and mmcv.VideoReader(str(output_path)).frame_cnt > 0
            ):
                bar.update(1)
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)

            out_frames = cutmix(action_path, scene_path, mask_bundle)

            if out_frames:
                frames_to_video(
                    out_frames,
                    output_path,
                    writer=conf.active.video.writer,
                    fps=fps,
                )

                n_written += 1
            else:
                print("out_frames None: ", filename)

            bar.update(1)

    bar.close()
    print("Written videos:", n_written)
    print("Skipped videos:", n_skipped)


if __name__ == "__main__":
    main()
