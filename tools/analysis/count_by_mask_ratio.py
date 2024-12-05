"""
This code counts the number of rejected videos by the specified mask ratio threshold.
"""

import sys

sys.path.append(".")

import json
from pathlib import Path

import click

from assertpy.assertpy import assert_that
from config import settings as conf


@click.command()
@click.argument(
    "mask-ratio",
    nargs=1,
    required=True,
    type=float,
)
def main(mask_ratio):
    ROOT = Path.cwd()
    DATASET = conf.active.dataset
    DETECTOR = conf.active.detector
    DET_CONF = str(conf.unidet.detect.confidence)
    METHOD = conf.active.mode
    USE_REPP = conf.active.USE_REPP
    mid_dir = ROOT / "data" / DATASET / DETECTOR / DET_CONF / METHOD

    if METHOD in ("allcutmix", "actorcutmix"):
        METHOD_DIR = mid_dir / ("REPP/mask" if USE_REPP else "mask")
    else:
        relevancy_method = conf.active.relevancy.method
        relevancy_thresh = str(conf.active.relevancy.threshold)

        METHOD_DIR = (
            mid_dir
            / ("REPP/mask" if USE_REPP else "mask")
            / relevancy_method
            / relevancy_thresh
        )

    assert_that(METHOD).is_in("allcutmix", "actorcutmix", "intercutmix")

    print("Mask:", METHOD_DIR.relative_to(ROOT))

    with open(METHOD_DIR / "ratio.json", "r") as file:
        data = json.load(file)

    n_excluded = sum(1 for k, v in data.items() if v < mask_ratio)
    percentage = round(n_excluded / len(data) * 100, 2)

    print("Excluded videos:", n_excluded, f"({percentage}%)")


if __name__ == "__main__":
    main()
