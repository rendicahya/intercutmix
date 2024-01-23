import json
from pathlib import Path

import cv2
from assertpy.assertpy import assert_that
from python_config import Config
from python_video import frames_to_video, video_frames, video_info
from tqdm import tqdm
from python_file import count_files

conf = Config("config.json")
input_root = Path(conf.repp.input.path)
datasets = {"ucf101": conf.ucf101.path, "hmdb51": conf.hmdb51.path}
video_root = Path(datasets[conf.repp.dataset])
output_video_root = Path(conf.repp.output.video.path)

assert_that(input_root).is_directory().is_readable()
assert_that(video_root).is_directory().is_readable()

n_files = count_files(input_root)
bar = tqdm(total=n_files)

for file in input_root.glob("**/*.json"):
    video_name = "_".join(file.name.split("_")[:4]) + ".avi"
    action = str(file).split('/')[-2]
    video_path = video_root / action / video_name

    assert_that(video_path).is_file().is_readable()
    bar.set_description(video_name)

    frames = video_frames(video_path, reader=conf.repp.video_reader)
    output_frames = []
    output_video_path = (output_video_root / action / video_name).with_suffix(".mp4")
    info = video_info(video_path)

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file) as f:
        data = json.load(f)

    for i, frame in enumerate(frames):
        boxes = [item["bbox"] for item in data if int(item["image_id"]) == i]

        for box in boxes:
            x1, y1, w, h = [round(v) for v in box]
            x2, y2 = x1 + w, y1 + h

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        output_frames.append(frame)

    frames_to_video(
        frames=output_frames,
        target=output_video_path,
        writer=conf.repp.output.video.writer,
        fps=info["fps"],
    )

    bar.update(1)
