import sys

sys.path.append(".")

import random
from pathlib import Path

import click
import mmcv
import numpy as np
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf
from python_video import frames_to_video

ROOT = Path.cwd()
DATASET = conf.active.DATASET
VIDEO_IN_DIR = ROOT / conf[DATASET].path
IN_EXT = conf[DATASET].ext
N_VIDEOS = conf[DATASET].N_VIDEOS
MULTIPLICATION = conf.cutmix.MULTIPLICATION
SEED = conf.active.random_seed

print("n videos:", N_VIDEOS)
print("Multiplication:", MULTIPLICATION)

assert_that(VIDEO_IN_DIR).is_directory().is_readable()

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

with open(VIDEO_IN_DIR / "list.txt", "r") as f:
    file_list = [line.split()[0] for line in f.readlines()]

random.seed(SEED)

bar = tqdm(total=N_VIDEOS * MULTIPLICATION, dynamic_ncols=True)

for line in file_list:
    path = Path(VIDEO_IN_DIR / line)
