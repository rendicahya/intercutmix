import sys

sys.path.append(".")

import json
from pathlib import Path

from assertpy.assertpy import assert_that
from config import settings as conf

k400_root = Path(conf.kinetics400.path)
ext = conf.kinetics400.ext

assert_that(k400_root).is_directory().is_readable()
print("Scanning directories...")

with open(k400_root / conf.kinetics400.file_list, "w") as f:
    for split in "train", "val", "test":
        for file in (k400_root / split).glob(f"**/*{ext}"):
            f.write(
                file.stem.strip() + " " + str(file.parent.relative_to(k400_root)) + "\n"
            )

with open(k400_root / conf.kinetics400.replacement_list, "w") as f:
    for file in (k400_root / "replacement").glob(f"**/*{ext}"):
        f.write(str(file) + "\n")
