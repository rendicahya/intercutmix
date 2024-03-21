import json
from pathlib import Path

from assertpy.assertpy import assert_that
from python_config import Config
from tqdm import tqdm

conf = Config("config.json")
relevancy_model = conf.relevancy.active.method
relevancy_threshold = conf.relevancy.active.threshold
video_root = Path(conf[conf.active.dataset].path).parent
min_mask_ratio = conf.cutmix.min_mask_ratio
mode = conf.active.mode
ext = conf.cutmix.output.ext

video_dir = (
    video_root
    / "REPP"
    / mode
    / "mix"
    / str(min_mask_ratio)
    / relevancy_model
    / str(relevancy_threshold)
)

assert_that(video_dir).is_directory().is_readable()

n_actions = conf[conf.active.dataset].n_classes
data = {}
bar = tqdm(total=n_actions)

for action in sorted(video_dir.iterdir()):
    files = [
        str(file.relative_to(video_dir))
        for file in action.iterdir()
        if file.is_file() and file.suffix == ext
    ]

    data[action.name] = files
    bar.update(1)

bar.close()

with open(video_dir / "list.json", "w") as f:
    json.dump(data, f, indent=2)
