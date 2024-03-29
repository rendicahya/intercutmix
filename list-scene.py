import json
from pathlib import Path

from assertpy.assertpy import assert_that
from config import settings as conf
from tqdm import tqdm

scene_dir = Path(conf.cutmix.input[conf.active.dataset].scene.path)
scene_ext = conf.cutmix.input[conf.active.dataset].scene.ext
json_out_path = conf.cutmix.input[conf.active.dataset].scene.list
n_actions = conf[conf.active.dataset].n_classes

assert_that(scene_dir).is_directory().is_readable()

data = {}
bar = tqdm(total=n_actions)

for action in sorted(scene_dir.iterdir()):
    files = [
        str(file.relative_to(scene_dir))
        for file in action.iterdir()
        if file.is_file() and file.suffix == scene_ext
    ]

    data[action.name] = files
    bar.update(1)

bar.close()

with open(json_out_path, "w") as f:
    json.dump(data, f, indent=2)
