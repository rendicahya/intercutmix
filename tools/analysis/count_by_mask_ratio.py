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
    root = Path.cwd()
    dataset = conf.active.dataset
    detector = conf.active.detector
    object_conf = str(conf.unidet.detect.confidence)
    method = conf.active.mode
    use_REPP = conf.active.use_REPP
    mid_dir = root / "data" / dataset / detector / object_conf / method

    if method in ("allcutmix", "actorcutmix"):
        mask_dir = mid_dir / ("REPP/mask" if use_REPP else "mask")
    else:
        relevancy_method = conf.active.relevancy.method
        relevancy_thresh = str(conf.active.relevancy.threshold)

        mask_dir = (
            mid_dir
            / ("REPP/mask" if use_REPP else "mask")
            / relevancy_method
            / relevancy_thresh
        )

    assert_that(method).is_in("allcutmix", "actorcutmix", "intercutmix")

    print("Mask:", mask_dir.relative_to(root))

    with open(mask_dir / "ratio.json", "r") as file:
        data = json.load(file)

    n_rejected = sum(1 for k, v in data.items() if v < mask_ratio)
    percentage = round(n_rejected / len(data) * 100, 2)

    print("Rejected videos:", n_rejected, f"({percentage}%)")


if __name__ == "__main__":
    main()
