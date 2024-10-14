import sys

sys.path.append(".")

import random
from pathlib import Path

from assertpy.assertpy import assert_that
from config import settings as conf


def save(file_paths, file):
    with open(file, "w") as f:
        for path in file_paths:
            f.write(str(path.relative_to(path.parent.parent)) + "\n")


def main():
    ROOT = Path.cwd()
    DATASET = conf.active.dataset
    DATASET_DIR = ROOT / conf[DATASET].path
    EXT = conf[DATASET].ext
    RANDOM_SEED = conf.active.random_seed
    TRAIN_RATIO = conf[DATASET].train_ratio

    assert_that(DATASET_DIR).is_directory().exists()

    file_paths = [path for path in DATASET_DIR.glob(f"**/*{EXT}")]

    random.seed(RANDOM_SEED)
    random.shuffle(file_paths)

    num_train = int(len(file_paths) * TRAIN_RATIO)
    train_set = file_paths[:num_train]
    test_set = file_paths[num_train:]

    save(train_set, DATASET_DIR.parent / f"{DATASET}_train_split_1_videos.txt")
    save(test_set, DATASET_DIR.parent / f"{DATASET}_val_split_1_videos.txt")


if __name__ == "__main__":
    main()
