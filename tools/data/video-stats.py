import sys

sys.path.append(".")

from collections import Counter
from pathlib import Path

from config import settings as conf
from prettytable import PrettyTable
from python_file import count_files
from python_video import video_info
from tqdm import tqdm

video_dir = Path(conf[conf.active.dataset].path)
ext = conf[conf.active.dataset].ext
counter = Counter()
bar = tqdm(total=count_files(video_dir))

for file in video_dir.glob(f"**/*{ext}"):
    info = video_info(file)
    w, h = info["width"], info["height"]
    counter[f"{w}×{h}"] += 1

    bar.update(1)

bar.close()

table = PrettyTable()
table.field_names = "Dimension", "Count"
table.reversesort = True

for dim, count in counter.items():
    table.add_row([dim, count])

print("Dataset:", conf.active.dataset)
print(table.get_string(sortby="Count"))
