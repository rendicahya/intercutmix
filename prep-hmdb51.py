from pathlib import Path

import numpy as np
from python_config import Config
from python_video import video_frames, video_info
from scipy.io import loadmat
import cv2

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
        # w, h = vid_info["width"], vid_info["height"]
        mat = loadmat(file)["part_mask"]
        mat_len = mat.shape[2]
        # cube = np.zeros((mat_len, h, w, 3), np.uint8)

        for f, frame in enumerate(frames):
            # cube[f] = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_path = (
                Path("data/hmdb51/frames") / action.name / file.stem / f"{f:05}.png"
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), frame)

            if f == mat_len - 1:
                break

        # np.savez_compressed(output_path, cube)

        # break
    break
