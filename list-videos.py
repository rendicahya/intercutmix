import json
from pathlib import Path

from assertpy.assertpy import assert_that
from config import settings as conf
from tqdm import tqdm

relevancy_model = conf.relevancy.active.method
relevancy_thresh = str(conf.relevancy.active.threshold)
dataset = conf.active.dataset
detector = conf.active.detector
mode = conf.active.mode
ext = conf.cutmix.output.ext
multiplication = conf.cutmix.multiplication
n_videos = conf[conf.active.dataset].n_videos * multiplication
n_actions = conf[conf.active.dataset].n_classes
use_REPP = conf.active.use_REPP
video_root = Path(conf[dataset].path).parent
object_selection = conf.active.object_selection

method = "select" if object_selection else "detect"
method_dir = Path("data") / dataset / detector / method
mix_part = "mix" if conf.random_seed is None else f"mix-{conf.random_seed}"
repp_part = f"REPP/{mix_part}" if use_REPP else mix_part

if method == "detect":
    video_dir = method_dir / repp_part
elif method == "select":
    video_dir = method_dir / mode / repp_part

    if mode == "intercutmix":
        video_dir = video_dir / relevancy_model / relevancy_thresh

print("Dataset:", dataset)
print("Mode:", mode)
print("REPP:", use_REPP)
print("Relevancy model:", relevancy_model)
print("Relevancy thresh.:", relevancy_thresh)
print("N videos:", n_videos)
print("Input:", video_dir)
print("Output:", video_dir / "list.json")

if not click.confirm("Do you want to continue?", show_default=True):
    exit("Aborted.")

assert_that(video_dir).is_directory().is_readable()

data = {}
bar = tqdm(total=n_actions)

for action in sorted(video_dir.iterdir()):
    if action.is_file():
        continue

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
