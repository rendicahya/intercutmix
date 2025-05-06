import sys

sys.path.append(".")

from collections import Counter
from pathlib import Path

from prettytable import PrettyTable
from tqdm import tqdm

from config import settings as conf
from python_file import count_files
from python_video import video_info

video_dir = Path(conf[conf.active.dataset].path)
ext = conf[conf.active.dataset].ext
counter = Counter()
durations = []
bar = tqdm(total=count_files(video_dir), dynamic_ncols=True)

for file in video_dir.glob(f"**/*{ext}"):
    info = video_info(file)
    w, h = info["width"], info["height"]
    counter[f"{w}×{h}"] += 1

    if "duration" in info:
        durations.append(info["duration"])

    bar.update(1)

bar.close()

table = PrettyTable()
table.field_names = "Dimension", "Count"
table.reversesort = True

for dim, count in counter.items():
    table.add_row([dim, count])

print("Dataset:", conf.active.dataset)
print("Video dimension counts:")
print(table.get_string(sortby="Count"))

min_duration = min(durations)
max_duration = max(durations)
average_duration = sum(durations) / len(durations)

print("\nVideo durations:")
print(f"  Minimum: {min_duration:.2f} s")
print(f"  Maximum: {max_duration:.2f} s")
print(f"  Average: {average_duration:.2f} s")
