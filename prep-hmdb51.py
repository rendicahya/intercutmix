from pathlib import Path

import numpy as np
from python_config import Config
from python_video import video_frames, video_info
from scipy.io import loadmat

conf = Config("config.json")
hmdb51_dir = Path(conf.hmdb51.path)
hmdb51_ext = conf.hmdb51.ext
mat_dir = Path(conf.hmdb51.mat.path)

for action in mat_dir.iterdir():
    for file in action.iterdir():
        video_path = hmdb51_dir / action.name / file.with_suffix(hmdb51_ext).name

        if not video_path.exists():
            print("Video not exist:", action.name + file.name)
            continue

        frames = video_frames(video_path)
        vid_info = video_info(video_path)
        mat = loadmat(file)["part_mask"]
        w, h = vid_info["width"], vid_info["height"]
        mat_len = mat.shape[2]
        cube = np.zeros((mat_len, h, w, 3), np.uint8)

        for f, frame in enumerate(frames):
            cube[f] = frame

            if f == mat_len - 1:
                break

        break
    break
