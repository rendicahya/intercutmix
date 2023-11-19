import json
from pathlib import Path
from tqdm import tqdm
from assert_utils import assert_dir, assert_file
from python_utils import Config

conf = Config("config.json")
assert_file("config.json", ".json")

scene_dir = Path(conf.mix.scene.path)
label_file = Path(conf.ucf101.class_index)

assert_dir(scene_dir)
assert_file(label_file)

data = {}
bar = tqdm(total=sum(1 for d in scene_dir.iterdir() if d.is_dir()))

for action in scene_dir.iterdir():
    files = [
        str(file.relative_to(scene_dir))
        for file in action.iterdir()
        if file.is_file() and file.suffix == conf.mix.scene.ext
    ]

    data[action.name] = files
    bar.update(1)

bar.close()

with open(conf.mix.scene.list, "w") as f:
    json.dump(data, f)
