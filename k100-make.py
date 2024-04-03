import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from assertpy.assertpy import assert_that
from config import settings as conf
from tqdm import tqdm

k400_dir = Path(conf.kinetics400.path)
k100_dir = Path(conf.kinetics100.path)
k400_filelist_path = k100_dir.parent / conf.kinetics400.filelist
k400_replacements = k100_dir.parent / conf.kinetics400.replacements.list
n_classes = conf.kinetics100.n_classes
partition = conf.kinetics100.partition
ext = conf.kinetics400.ext
op = conf.kinetics100.make
replacement_dir = conf.kinetics400.replacements.dir
replacement_count = 0
report = []
n_files = defaultdict(int)

assert_that(k400_dir).is_directory().is_readable()
assert_that(k400_filelist_path).is_file().is_readable()
assert_that(k400_replacements).is_file().is_readable()

with open(k400_filelist_path) as f:
    k400_filelist = json.load(f)

with open(k400_replacements) as f:
    replacements = json.load(f)

for split in "labeled0", "unlabeled0", "val0":
    file_list_path = (
        Path("VideoSSL/datasplit/kinetics")
        / f"ssl_sub{n_classes}c_{partition}_{split}.lst"
    )

    assert_that(file_list_path).is_file().is_readable()
    print(f"\nSplit: {split}")

    with open(file_list_path) as f:
        file_list = f.readlines()

    bar = tqdm(total=len(file_list))

    for file in file_list:
        _, action, filename = file.strip().split("/")
        filename = filename.split("*")[0]
        stem = filename.split(".")[0]
        dst = k100_dir.parent / split / action / filename
        n_files[split] += 1

        if stem in replacements:
            src = k400_dir / replacement_dir / filename
            replacement_count += 1
        elif stem in k400_filelist:
            src = k400_dir / k400_filelist[stem] / filename
        else:
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        bar.update(1)

        if op == "copy":
            shutil.copy(src, dst)
        else:
            os.symlink(src, dst)

    bar.close()

with open(k100_dir.parent / "report.txt", "w") as f:
    f.write(f"Replacements: {replacement_count}\n")

    for split in "labeled0", "unlabeled0", "val0":
        f.write(f"{split}: {n_files[split]}\n")
