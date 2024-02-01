from pathlib import Path

import numpy as np
from assertpy.assertpy import assert_that
from python_config import Config
from python_file import count_files
from scipy.io import loadmat
from tqdm import tqdm

if __name__ == "__main__":
    conf = Config("config.json")
    mat_dir = Path(conf.hmdb51.mat.path)
    hmdb51_dir = Path(conf.hmdb51.path)
    hmdb51_ext = conf.hmdb51.ext
    out_dir = Path(conf.hmdb51.mat.mask.path)
    bar = tqdm(total=count_files(mat_dir))

    assert_that(mat_dir).is_directory().is_readable()

    for file in mat_dir.glob("**/*.mat"):
        stem = file.name.split(".")[0]

        bar.set_description(stem[:30])

        action = file.parent.name
        mat = loadmat(file)
        # Change from (h, w, t) to (t, h, w)
        mask_cube = np.moveaxis(mat["part_mask"], -1, 0)
        mask_cube *= 255
        out_path = out_dir / action / stem

        out_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(out_path, mask_cube)
        bar.update(1)

    bar.close()
