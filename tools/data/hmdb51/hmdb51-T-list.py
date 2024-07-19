import sys

sys.path.append(".")

import os
from pathlib import Path

from config import settings as conf

hmdb51_dir = Path(conf.hmdb51.path)
hmdb51_T_dir = Path(conf["hmdb51-T"].path)
ext = conf.hmdb51.ext
hmdb51_T_index = [file.stem for file in hmdb51_T_dir.glob(f"**/*{ext}")]

for split in "train", "val":
    for split_no in range(1, 4):
        hmdb51_T_list = []

        with open(
            hmdb51_dir.parent / f"hmdb51_{split}_split_{split_no}_videos.txt", "r"
        ) as f:
            for line in f:
                path, class_id = line.strip().split()
                action, filename = path.split("/")
                stem, e = filename.split(".")

                if stem in hmdb51_T_index:
                    hmdb51_T_list.append(line)

        with open(
            hmdb51_T_dir.parent / f"hmdb51-T_{split}_split_{split_no}_videos.txt", "w"
        ) as f:
            f.write("".join(hmdb51_T_list))
