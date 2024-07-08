import sys

sys.path.append(".")

import json
from os import symlink
from pathlib import Path
from shutil import copy2

import click
from config import settings as conf


@click.command()
@click.argument(
    "actorcutmix-mask-dir",
    nargs=1,
    required=True,
    type=click.Path(
        exists=True,
        readable=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.argument(
    "intercutmix-mask-dir",
    nargs=1,
    required=True,
    type=click.Path(
        exists=True,
        readable=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
def main(actorcutmix_mask_dir, intercutmix_mask_dir):
    with open(actorcutmix_mask_dir / "ratio.json") as f:
        acm_ratio_json = json.load(f)

    with open(intercutmix_mask_dir / "ratio.json") as f:
        icm_ratio_json = json.load(f)

    dataset = conf.active.dataset
    dataset_dir = Path(conf[dataset].path)
    dataset_T_dir = Path(conf[f"{dataset}-T"].path)
    ext = conf[dataset].ext
    file_index = {}
    count = 0

    for file in dataset_dir.glob(f"**/*{ext}"):
        action = file.parent.name

        file_index.update({file.stem: action})

    for file in acm_ratio_json.keys():
        acm_ratio = acm_ratio_json[file]
        icm_ratio = icm_ratio_json[file]

        if icm_ratio <= acm_ratio:
            continue

        action = file_index[file]
        src = dataset_dir / action / (file + ext)
        dst = dataset_T_dir / action / (file + ext)
        count += 1

        dst.parent.mkdir(parents=True, exist_ok=True)
        symlink(src, dst)

    print(f"Built {count} files.")


if __name__ == "__main__":
    main()
