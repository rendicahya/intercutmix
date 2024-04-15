import json
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
subdirs = set()
files = []

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

        if stem in replacements:
            src = k400_dir / replacement_dir / filename
        elif stem in k400_filelist:
            src = k400_dir / k400_filelist[stem] / filename
        else:
            continue

        subdirs.add(f"{split}/{action}")
        files.append(src)
        bar.update(1)

    bar.close()

with open(k100_dir.parent / "remote-copy.sh", "w") as f:
    for subdir in subdirs:
        f.write(
            f'ssh randy@jupiter "mkdir -p /home/randy/datasets/kinetics100/{subdir}"\n'
        )

    f.write(f"scp -P 2200 -r {src} randy@jupiter:/home/randy/datasets/kinetics100/")
