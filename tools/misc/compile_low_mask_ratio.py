import sys

sys.path.append(".")

from pathlib import Path

import click
from config import settings as conf

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

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")
