import sys

sys.path.append(".")
import json
from pathlib import Path

import numpy as np
from assertpy.assertpy import assert_that
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

if method == "detect":
    mask_dir = method_dir / ("REPP/mask" if use_REPP else "mask")
elif method == "select":
    mask_dir = method_dir / mode / ("REPP/mask" if use_REPP else "mask")

    if mode == "intercutmix":
        mask_dir = mask_dir / relevancy_model / relevancy_thresh

ratio_file = mask_dir / "ratio.json"

assert_that(ratio_file).is_file().is_readable()

with open(ratio_file, "r") as file:
    ratio_data = json.load(file)

average_ratio = sum([ratio for file, ratio in ratio_data.items()]) / len(ratio_data)

print("File:", ratio_file)
print("Average ratio:", round(average_ratio * 100, 2))
