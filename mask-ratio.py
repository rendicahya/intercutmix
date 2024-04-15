import json
from pathlib import Path

import click
import numpy as np
from assertpy.assertpy import assert_that
from config import settings as conf
from tqdm import tqdm

dataset = conf.active.dataset
detector = conf.active.detector
mode = conf.active.mode
relevancy_model = conf.relevancy.active.method
relevancy_thresh = str(conf.relevancy.active.threshold)
n_files = conf[conf.active.dataset].n_videos
use_REPP = conf.active.use_REPP
object_selection = conf.active.object_selection

method = "select" if object_selection else "detect"
method_dir = Path("data") / dataset / detector / method

if method == "detect":
    mask_dir = method_dir / ("REPP/mask" if use_REPP else "mask")
elif method == "select":
    mask_dir = method_dir / mode / ("REPP/mask" if use_REPP else "mask")

    if mode == "intercutmix":
        mask_dir = mask_dir / relevancy_model / relevancy_thresh

out_path = mask_dir / "ratio.json"

print("Dataset:", dataset)
print("Mode:", mode)
print("REPP:", conf.active.use_REPP)
print("Relevancy model:", relevancy_model)
print("Relevancy threshold:", relevancy_thresh)
print("Σ videos:", n_files)
print("Input:", mask_dir)
print("Output:", out_path)

if not click.confirm("Do you want to continue?", show_default=True):
    exit("Aborted.")

assert_that(mask_dir).is_directory().is_readable()

data = {}
bar = tqdm(total=n_files)

for mask_path in mask_dir.glob("**/*.npz"):
    mask_bundle = np.load(mask_path)["arr_0"]
    mask_ratio = round(np.count_nonzero(mask_bundle) / mask_bundle.size, 3)
    data[mask_path.stem] = mask_ratio

    bar.update(1)

bar.close()

with open(out_path, "w") as f:
    json.dump(data, f)
