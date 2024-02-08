import json
from pathlib import Path

from assertpy.assertpy import assert_that
from python_config import Config
from tqdm import tqdm

conf = Config("config.json")
scene_dir = Path(conf.cutmix.input.scene.path)

assert_that(scene_dir).is_directory().is_readable()

n_actions = sum(1 for d in scene_dir.iterdir() if d.is_dir())
data = {}
bar = tqdm(total=n_actions)

for action in sorted(scene_dir.iterdir()):
    files = [
        str(file.relative_to(scene_dir))
        for file in action.iterdir()
        if file.is_file() and file.suffix == conf.cutmix.input.scene.ext
    ]

    data[action.name] = files
    bar.update(1)

bar.close()

with open(conf.cutmix.scene.list, "w") as f:
    json.dump(data, f, indent=2)
