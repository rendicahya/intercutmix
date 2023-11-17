import json
from pathlib import Path

from utils import Config, assert_dir, assert_file

conf = Config("config.json")
assert_file("config.json", ".json")

scene_dir = Path(conf.mix.scene.path)
label_file = Path(conf.ucf101.class_index)

assert_dir(scene_dir)
assert_file(label_file)

data = {}

for action in scene_dir.iterdir():
    files = [
        str(file.name)
        for file in action.iterdir()
        if file.is_file() and file.suffix == conf.mix.scene.ext
    ]

    data[action.name] = files

with open(conf.mix.scene.list, "w") as f:
    json.dump(data, f)
