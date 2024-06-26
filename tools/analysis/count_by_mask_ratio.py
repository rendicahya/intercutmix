"""
This code counts the number of rejected videos by the specified mask ratio threshold.
"""

import sys

sys.path.append(".")

import json
from pathlib import Path

import click
from config import settings as conf


@click.command()
@click.argument(
    "mask-ratio",
    nargs=1,
    required=True,
    type=float,
)
def main(mask_ratio):
    dataset = conf.active.dataset
    detector = conf.active.detector
    object_selection = conf.active.object_selection
    mode = conf.active.mode
    use_REPP = conf.active.use_REPP
    relevancy_model = conf.relevancy.active.method
    relevancy_thresh = str(conf.relevancy.active.threshold)
    method = "select" if object_selection else "detect"
    method_dir = Path("data") / dataset / detector / method

    print("Dataset:", dataset)
    print("Mode:", mode)

    if method == "detect":
        mask_in_dir = method_dir / ("REPP/mask" if use_REPP else "mask")
    elif method == "select":
        mask_in_dir = method_dir / mode / ("REPP/mask" if use_REPP else "mask")

        if mode == "intercutmix":
            mask_in_dir = mask_in_dir / relevancy_model / relevancy_thresh

    with open(mask_in_dir / "ratio.json", "r") as file:
        data = json.load(file)

    n_rejected = sum(1 for k, v in data.items() if v < mask_ratio)
    percentage = round(n_rejected / len(data) * 100, 2)

    print("Rejected videos:", n_rejected, f"({percentage}%)")


if __name__ == "__main__":
    main()
