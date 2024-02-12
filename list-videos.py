import json
from pathlib import Path

from assertpy.assertpy import assert_that
from python_config import Config
from tqdm import tqdm

conf = Config("config.json")
video_root = Path(conf.cutmix.output.path).parent

for mode in "actorcutmix", "intercutmix":
    video_dir = video_root / mode

    assert_that(video_dir).is_directory().is_readable()

    n_actions = sum(1 for d in video_dir.iterdir() if d.is_dir())
    data = {}
    bar = tqdm(total=n_actions)

    for action in sorted(video_dir.iterdir()):
        files = [
            str(file.relative_to(video_dir))
            for file in action.iterdir()
            if file.is_file() and file.suffix == conf.cutmix.input.scene.ext
        ]

        data[action.name] = files
        bar.update(1)

    bar.close()

    with open(video_root / f"{mode}.json", "w") as f:
        json.dump(data, f, indent=2)
