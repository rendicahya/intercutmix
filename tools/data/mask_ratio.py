import sys

sys.path.append(".")

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf


def calc_ratio(path):
    mask = np.load(path)["arr_0"]
    ratio = np.count_nonzero(mask) / mask.size

    return path.stem, round(ratio, 4)


@click.command()
@click.argument(
    "mask_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
def main(mask_dir):
    DATASET = conf.active.dataset
    N_FILES = conf[DATASET].n_videos
    MAX_WORKERS = conf.active.threads
    OUT_PATH = mask_dir / "ratio.json"
    data = {}

    print("Input:", mask_dir)
    print("Output:", OUT_PATH)
    print("n videos:", N_FILES)

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    assert_that(mask_dir).is_directory().is_readable()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exec:
        jobs = {
            exec.submit(calc_ratio, path): path for path in mask_dir.glob("**/*.npz")
        }

        for future in tqdm(as_completed(jobs), total=N_FILES, dynamic_ncols=True):
            path_stem, ratio = future.result()
            data[path_stem] = ratio

    with open(OUT_PATH, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
