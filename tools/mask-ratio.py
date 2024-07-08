import sys

sys.path.append(".")

import json
from pathlib import Path

import click
import numpy as np
from assertpy.assertpy import assert_that
from config import settings as conf
from tqdm import tqdm

dataset = conf.active.dataset
detector = conf.active.detector
object_selection = conf.active.object_selection
mode = conf.active.mode
use_REPP = conf.active.use_REPP
relevancy_model = conf.active.relevancy.method
relevancy_thresh = str(conf.active.relevancy.threshold)
n_files = conf[dataset].n_videos

method = "select" if object_selection else "detect"
method_dir = Path("data") / dataset / detector / method

if method == "detect":
    mask_in_dir = method_dir / ("REPP/mask" if use_REPP else "mask")
elif method == "select":
    mask_in_dir = method_dir / mode / ("REPP/mask" if use_REPP else "mask")

    if mode == "intercutmix":
        mask_in_dir = mask_in_dir / relevancy_model / relevancy_thresh

json_out_path = mask_in_dir / "ratio.json"

print("Input:", mask_in_dir)
print("Output:", json_out_path)
print("Σ videos:", n_files)

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

assert_that(mask_in_dir).is_directory().is_readable()

data = {}
bar = tqdm(total=n_files)

for mask_path in mask_in_dir.glob("**/*.npz"):
    mask_bundle = np.load(mask_path)["arr_0"]
    mask_ratio = np.count_nonzero(mask_bundle) / mask_bundle.size
    data[mask_path.stem] = mask_ratio

    bar.update(1)

bar.close()

with open(json_out_path, "w") as f:
    json.dump(data, f)
