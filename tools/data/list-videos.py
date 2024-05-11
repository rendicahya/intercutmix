import sys

sys.path.append(".")

import os
from pathlib import Path

import click
from assertpy.assertpy import assert_that
from config import settings as conf
from python_file import count_files
from tqdm import tqdm


@click.command()
@click.argument(
    "click-path",
    nargs=1,
    required=False,
    type=click.Path(
        exists=True,
        readable=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
def main(click_path):
    dataset = conf.active.dataset
    detector = conf.active.detector
    mode = conf.active.mode
    random_seed = conf.active.random_seed
    object_selection = conf.active.object_selection
    use_REPP = conf.active.use_REPP
    relevancy_model = conf.relevancy.active.method
    relevancy_thresh = str(conf.relevancy.active.threshold)

    if click_path:
        video_in_dir = click_path
    else:
        method = "select" if object_selection else "detect"
        method_dir = Path("data") / dataset / detector / method
        mix_mode = "mix" if random_seed is None else f"mix-{random_seed}"

        if method == "detect":
            video_in_dir = method_dir / (f"REPP/{mix_mode}" if use_REPP else mix_mode)
        elif method == "select":
            video_in_dir = (
                method_dir / mode / (f"REPP/{mix_mode}" if use_REPP else mix_mode)
            )

            if mode == "intercutmix":
                video_in_dir = video_in_dir / relevancy_model / relevancy_thresh

    assert_that(video_in_dir).is_directory().is_readable()

    out_file_path = video_in_dir / "list.txt"
    n_videos = count_files(video_in_dir)

    print("Input:", video_in_dir)
    print("Output:", out_file_path)
    print("Σ videos:", n_videos)

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    data = []
    bar = tqdm(total=n_videos)
    class_id = 0

    for action in sorted(video_in_dir.iterdir()):
        if action.is_file():
            continue

        for file in sorted(action.iterdir()):
            if file.is_dir():
                continue

            line = f"{file.relative_to(video_in_dir)} {class_id}"

            data.append(line)
            bar.update(1)

        class_id += 1

    bar.close()

    with open(out_file_path, "w") as f:
        f.write(os.linesep.join(data))


if __name__ == "__main__":
    main()
