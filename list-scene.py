import json
from pathlib import Path

import click
from assertpy.assertpy import assert_that
from config import settings as conf
from tqdm import tqdm

dataset = conf.active.dataset
scene_in_dir = Path(conf.cutmix.input[dataset].scene.path)
scene_ext = conf.cutmix.input[dataset].scene.ext
json_out_path = conf.cutmix.input[dataset].scene.list
n_actions = conf[dataset].n_classes

print("Dataset:", dataset)
print("Input:", mask_in_dir)
print("Output:", video_out_dir)

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

assert_that(scene_in_dir).is_directory().is_readable()

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
