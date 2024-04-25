import sys

sys.path.append(".")

from pathlib import Path

import cv2
import numpy as np
from assertpy.assertpy import assert_that
from config import settings as conf
from python_file import count_files
from scipy.io import loadmat
from tqdm import tqdm

mat_dir = Path(conf.hmdb51.mat.path)
hmdb51_dir = Path(conf.hmdb51.path)
hmdb51_ext = conf.hmdb51.ext
out_dir = Path(conf.hmdb51.mat.mask.path)
bar = tqdm(total=count_files(mat_dir))

assert_that(hmdb51_dir).is_directory().is_readable()
assert_that(mat_dir).is_directory().is_readable()

for file in mat_dir.glob("**/*.mat"):
    stem = file.name.split(".")[0]
    action = file.parent.name
    mat = loadmat(file)
    # Change from (h, w, t) to (t, h, w)
    mask_cube = np.moveaxis(mat["part_mask"], -1, 0)
    mask_cube *= 255

    for f, mask in enumerate(mask_cube):
        out_path = out_dir / action / stem / ("%05d.png" % f)

        if conf.hmdb51.mat.mask.box:
            x, y, w, h = cv2.boundingRect(mask)
            black = np.zeros((*mask.shape, 3), np.uint8)
            mask = cv2.rectangle(black, (x, y), (x + w, y + h), (255, 255, 255), -1)

        out_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(out_path), mask)

    bar.update(1)

bar.close()
