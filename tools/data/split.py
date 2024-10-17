import sys

sys.path.append(".")

import random
from pathlib import Path

from assertpy.assertpy import assert_that
from config import settings as conf


def save(file_paths, file, class_ind):
    with open(file, "w") as f:
        for path in file_paths:
            action = path.parent.name
            action_no = class_ind[action] - 1
            file = path.name

            f.write(f"{action}/{file} {action_no}\n")


def main():
    ROOT = Path.cwd()
    DATASET = conf.active.dataset
    DATASET_DIR = ROOT / conf[DATASET].path
    EXT = conf[DATASET].ext
    RANDOM_SEED = conf.active.random_seed
    TRAIN_RATIO = conf[DATASET].train_ratio
    CLASS_IND_FILE = DATASET_DIR.parent / "annotations/classInd.txt"
    CLASS_IND = {}

    assert_that(DATASET_DIR).is_directory().exists()
    assert_that(CLASS_IND_FILE).is_file().exists()

    with open(CLASS_IND_FILE, "r") as f:
        for line in f:
            number, action = line.strip().split()
            CLASS_IND[action] = int(number)

    file_paths = [path for path in DATASET_DIR.glob(f"**/*{EXT}")]

    random.seed(RANDOM_SEED)
    random.shuffle(file_paths)

    num_train = int(len(file_paths) * TRAIN_RATIO)
    train_set = file_paths[:num_train]
    test_set = file_paths[num_train:]

    train_file = DATASET_DIR.parent / f"{DATASET}_train_split_1_videos.txt"
    test_file = DATASET_DIR.parent / f"{DATASET}_val_split_1_videos.txt"

    save(train_set, train_file, CLASS_IND)
    save(test_set, test_file, CLASS_IND)


if __name__ == "__main__":
    main()
