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
    video_in_dir = click_path if click_path else Path(conf[dataset].path)
    out_file_path = video_in_dir / "list.txt"
    n_videos = count_files(video_in_dir)

    assert_that(video_in_dir).is_directory().is_readable()

    print("Dataset:", dataset)
    print("Σ videos:", n_videos)
    print("Input:", video_in_dir)
    print("Output:", out_file_path)

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
