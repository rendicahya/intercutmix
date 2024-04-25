import sys

sys.path.append(".")

import json
from pathlib import Path

import click
from assertpy.assertpy import assert_that
from config import settings as conf
from python_file import count_files
from tqdm import tqdm

dataset = conf.active.dataset
scene_in_dir = Path(conf[dataset].scene.path)
scene_ext = conf[dataset].scene.ext
json_out_path = scene_in_dir / "list.json"
n_actions = conf[dataset].n_classes
n_input = count_files(scene_in_dir, ext=scene_ext)

assert_that(scene_in_dir).is_directory().is_readable()

print("Dataset:", dataset)
print("Σ input:", n_input)
print("Input:", scene_in_dir)
print("Output:", json_out_path)

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

data = {}
bar = tqdm(total=n_actions)

for action in sorted(scene_in_dir.iterdir()):
    files = [
        str(file.relative_to(scene_in_dir))
        for file in action.iterdir()
        if file.is_file() and file.suffix == scene_ext
    ]

    data[action.name] = files
    bar.update(1)

bar.close()

with open(json_out_path, "w") as f:
    json.dump(data, f, indent=2)
