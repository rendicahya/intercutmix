import json
from pathlib import Path

import numpy as np
from assertpy.assertpy import assert_that
from config import settings as conf
from tqdm import tqdm

dataset = conf.active.dataset
mode = conf.active.mode
relevancy_model = conf.relevancy.active.method
relevancy_threshold = conf.relevancy.active.threshold
n_files = conf[conf.active.dataset].n_videos

if conf.cutmix.use_REPP:
    mode_dir = Path("data") / dataset / "REPP" / mode
else:
    mode_dir = Path("data") / dataset / detector / "select" / mode

mask_dir = mode_dir / "mask" / relevancy_model / str(relevancy_threshold)
out_path = mask_dir / "ratio.json"

print("Dataset:", dataset)
print("Mode:", mode)
print("REPP:", conf.cutmix.use_REPP)
print("Relevancy model:", relevancy_model)
print("Relevancy thresh.:", relevancy_threshold)
print("N videos:", n_files)
print("Input:", mask_dir)
print("Output:", out_path)

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
