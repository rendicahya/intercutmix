import sys

sys.path.append(".")

import os
import shutil
from collections import defaultdict
from pathlib import Path

from assertpy.assertpy import assert_that
from config import settings as conf
from tqdm import tqdm

k400_dir = Path.cwd() / conf.kinetics400.path
k100_dir = Path.cwd() / conf.kinetics100.path
k400_filelist_path = k400_dir / conf.kinetics400.file_list
k400_replacements = k400_dir / conf.kinetics400.replacement_list
n_classes = conf.kinetics100.n_classes
partition = conf.kinetics100.partition
ext = conf.kinetics400.ext
op = conf.kinetics100.make
n_files = defaultdict(int)
k400_filelist = {}
report = []
replacement_count = 0
not_found_count = 0

assert_that(k400_dir).is_directory().is_readable()
assert_that(k400_filelist_path).is_file().is_readable()
assert_that(k400_replacements).is_file().is_readable()

with open(k400_filelist_path) as file:
    for line in file:
        file_name, dir = line.split()
        k400_filelist[file_name] = dir

with open(k400_replacements) as file:
    replacements = file.readlines()

for part in "labeled0", "unlabeled0", "val0":
    file_list_path = (
        Path("VideoSSL/datasplit/kinetics")
        / f"ssl_sub{n_classes}c_{partition}_{part}.lst"
    )

    assert_that(file_list_path).is_file().is_readable()
    print(f"\nPart: {part}")

    with open(file_list_path) as file:
        file_list = file.readlines()

    bar = tqdm(total=len(file_list), dynamic_ncols=True)

    for file in file_list:
        split, action, filename = file.strip().split("/")
        filename = filename.split("*")[0]
        stem = filename.split(".")[0]

        if split == "Val":
            split = "test"

        dst = k100_dir / action / filename
        n_files[part] += 1

        if stem in replacements:
            src = k400_dir / "replacement" / filename
            replacement_count += 1
        elif stem in k400_filelist:
            src = k400_dir / k400_filelist[stem] / filename
        else:
            not_found_count += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        bar.update(1)

        if op == "copy":
            shutil.copy(src, dst)
        elif op == "symlink":
            os.symlink(src, dst)

    bar.close()

with open(k100_dir.parent / "report.txt", "w") as f:
    f.write(f"Replacements: {replacement_count}\n")
    f.write(f"Not found: {not_found_count}\n")

    for part in "labeled0", "unlabeled0", "val0":
        f.write(f"{part}: {n_files[part]}\n")
