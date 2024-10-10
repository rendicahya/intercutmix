import sys

sys.path.append(".")

import json
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from assertpy.assertpy import assert_that


@click.command()
@click.argument(
    "mask-dir",
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
def main(mask_dir):
    n_videos = sum(1 for f in mask_dir.glob("**/*.*"))
    json_out_path = mask_dir / "ratio.json"

    print("Input:", mask_dir)
    print("Output:", json_out_path)
    print("n videos:", n_videos)

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    assert_that(mask_dir).is_directory().is_readable()

    data = {}
    bar = tqdm(total=n_videos, dynamic_ncols=True)

    for mask_path in mask_dir.glob("**/*.npz"):
        mask_bundle = np.load(mask_path)["arr_0"]
        mask_ratio = np.count_nonzero(mask_bundle) / mask_bundle.size
        data[mask_path.stem] = round(mask_ratio, 4)

        bar.update(1)

    bar.close()

    with open(json_out_path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
