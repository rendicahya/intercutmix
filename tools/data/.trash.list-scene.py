import sys

sys.path.append(".")

import os
from pathlib import Path

import click
from assertpy.assertpy import assert_that
from config import settings as conf
from python_file import count_files
from tqdm import tqdm

dataset = conf.active.dataset
video_in_dir = Path(conf[dataset].scene.path)
video_ext = conf[dataset].scene.ext
file_out_path = video_in_dir / "list.txt"
n_videos = count_files(video_in_dir, ext=video_ext)

assert_that(video_in_dir).is_directory().is_readable()

print("Dataset:", dataset)
print("Σ input:", n_videos)
print("Input:", video_in_dir)
print("Output:", file_out_path)

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

data = []
bar = tqdm(total=n_videos, dynamic_ncols=True)
class_id = 0

for action in sorted(video_in_dir.iterdir()):
    if action.is_file():
        continue

    for file in sorted(action.iterdir()):
        if file.is_dir() or file.suffix != video_ext:
            continue

        line = f"{file.relative_to(video_in_dir)} {class_id}"

        data.append(line)
        bar.update(1)

    class_id += 1

bar.close()

with open(file_out_path, "w") as f:
    f.write(os.linesep.join(data))
