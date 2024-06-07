"""
This code counts the number of transitive and intransitive actions, i.e. action with and without involving objects, by comparing mask ratios.
"""

import sys

sys.path.append(".")
import json
from pathlib import Path

import click


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

    transitive_count = 0

    for key in acm_ratio_json.keys():
        acm_ratio = acm_ratio_json[key]
        icm_ratio = icm_ratio_json[key]

        if icm_ratio > acm_ratio:
            transitive_count += 1

    transitive_percent = round(transitive_count / len(acm_ratio_json) * 100, 2)

    print("Total:", len(acm_ratio_json))
    print(f"Transitive: {transitive_count} ({transitive_percent}%)")


if __name__ == "__main__":
    main()
