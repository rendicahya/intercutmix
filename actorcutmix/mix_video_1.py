from pathlib import Path

import numpy as np
from moviepy.editor import ImageSequenceClip, VideoFileClip
from PIL import Image

if __name__ == "__main__":
    video1_path = Path(
        "/nas.dbms/randy/datasets/ucf101/Basketball/v_Basketball_g01_c01.avi"
    )
    video2_path = Path(
        "/nas.dbms/randy/datasets/ucf101/Basketball/v_Basketball_g05_c01.avi"
    )
    mask_base_path = Path("/nas.dbms/randy/datasets/ucf101-bbox-mask")
    mask1_path = mask_base_path / video1_path.parent.name / video1_path.stem
    mask2_path = mask_base_path / video2_path.parent.name / video2_path.stem

    assert video1_path.exists() and video1_path.is_file()
    assert video2_path.exists() and video2_path.is_file()
    assert mask_base_path.exists() and mask_base_path.is_dir()
    assert mask1_path.exists() and mask1_path.is_dir()
    assert mask2_path.exists() and mask2_path.is_dir()

    video1 = VideoFileClip(str(video1_path))
    video1_mask = [Image.open(file).convert("L") for file in mask1_path.iterdir()]

    video2 = VideoFileClip(str(video2_path))
    video2_mask = [Image.open(file).convert("L") for file in mask2_path.iterdir()]

    n_frames = min(video1.reader.nframes, video2.reader.nframes) - 1

    video1_frames = list(video1.iter_frames())[:n_frames]
    video1_mask = video1_mask[:n_frames]

    video2_frames = list(video2.iter_frames())[:n_frames]
    video2_mask = video2_mask[:n_frames]

    mix1 = []
    mix2 = []

    black1 = Image.new("L", video1_mask[0].size)
    black2 = Image.new("L", video2_mask[0].size)

    for f in range(n_frames):
        image1 = Image.fromarray(video1_frames[f])
        mask1 = video1_mask[f]
        scene1 = Image.composite(black1, image1, mask1)

        image2 = Image.fromarray(video2_frames[f])
        mask2 = video2_mask[f]
        scene2 = Image.composite(black2, image2, mask2)

        composite1 = Image.composite(image1, scene2, mask1)
        composite2 = Image.composite(image2, scene1, mask2)

        mix1.append(np.asarray(composite1))
        mix2.append(np.asarray(composite2))

    ImageSequenceClip(mix1, fps=video1.fps).write_videofile("mix1.mp4", audio=False)
    ImageSequenceClip(mix2, fps=video2.fps).write_videofile("mix2.mp4", audio=False)
