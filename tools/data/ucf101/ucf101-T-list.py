import sys

sys.path.append(".")

import os
from pathlib import Path

from config import settings as conf

ucf101_dir = Path(conf.ucf101.path)
ucf101_T_dir = Path(conf["ucf101-T"].path)
ext = conf.ucf101.ext
ucf101_T_index = [file.stem for file in ucf101_T_dir.glob(f"**/*{ext}")]

for split in "train", "val":
    for split_no in range(1, 4):
        ucf101_T_list = []
        count = 0

        with open(
            ucf101_dir.parent / f"ucf101_{split}_split_{split_no}_videos.txt", "r"
        ) as f:
            for line in f:
                path, class_id = line.strip().split()
                action, filename = path.split("/")
                stem, e = filename.split(".")

                if stem in ucf101_T_index:
                    ucf101_T_list.append(line)
                    count += 1

        with open(
            ucf101_T_dir.parent / f"ucf101-T_{split}_split_{split_no}_videos.txt", "w"
        ) as f:
            f.write("".join(ucf101_T_list))
