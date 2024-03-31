import json
from pathlib import Path

from assertpy.assertpy import assert_that
from config import settings as conf

k400_dir = Path(conf.kinetics400.path)
k400_replacement_dir = k400_dir / conf.kinetics400.replacements.dir
k100_dir = Path(conf.kinetics100.path)
ext = conf.kinetics400.ext
files = {}

assert_that(k400_dir).is_directory().is_readable()
assert_that(k400_replacement_dir).is_directory().is_readable()

for split in "train", "val", "test":
    for file in (k400_dir / split).glob(f"**/*{ext}"):
        files[file.stem.strip()] = str(file.parent.relative_to(k400_dir))

k100_dir.parent.mkdir(parents=True, exist_ok=True)

with open(k100_dir.parent / conf.kinetics400.filelist, "w") as f:
    json.dump(files, f)

replacements = [file.stem for file in k400_replacement_dir.glob(f"**/*{ext}")]

with open(k100_dir.parent / conf.kinetics400.replacements.list, "w") as f:
    json.dump(replacements, f)
