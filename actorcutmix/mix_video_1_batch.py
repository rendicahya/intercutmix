import pathlib
import random

import click
import numpy as np
from moviepy.editor import ImageSequenceClip, VideoFileClip
from PIL import Image
from tqdm import tqdm


@click.command()
@click.argument(
    "file-list-path",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    "actor-video-path",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    "scene-video-path",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    "mask-path",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    "output-path",
    nargs=1,
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
def main(file_list_path, actor_video_path, scene_video_path, mask_path, output_path):
    file_paths = open(file_list_path, "r").readlines()

    for file1 in tqdm(file_paths):
        file1 = file1.strip()

        file2 = random.choice(file_paths)
        file2 = file2.strip()

        video1_path = actor_video_path / file1
        video2_path = actor_video_path / file2

        video1_mask_path = mask_path / file1.split(".")[0]
        video2_mask_path = mask_path / file2.split(".")[0]

        scene1_path = scene_video_path / file1
        scene2_path = scene_video_path / file2

        video1 = VideoFileClip(str(video1_path))
        scene1 = VideoFileClip(str(scene1_path))
        mask1 = [Image.open(file).convert("L") for file in video1_mask_path.iterdir()]

        video2 = VideoFileClip(str(video2_path))
        scene2 = VideoFileClip(str(scene2_path))
        mask2 = [Image.open(file).convert("L") for file in video2_mask_path.iterdir()]

        n_frames = min(video1.reader.nframes, video2.reader.nframes) - 1

        video1_frames = list(video1.iter_frames())[:n_frames]
        scene1_frames = list(scene1.iter_frames())[:n_frames]
        mask1 = mask1[:n_frames]

        video2_frames = list(video2.iter_frames())[:n_frames]
        scene2_frames = list(scene2.iter_frames())[:n_frames]
        mask2 = mask2[:n_frames]

        mix1 = []
        mix2 = []

        for f in range(n_frames):
            video1_frame = Image.fromarray(video1_frames[f])
            scene1_frame = Image.fromarray(scene1_frames[f])

            video2_frame = Image.fromarray(video2_frames[f])
            scene2_frame = Image.fromarray(scene2_frames[f])

            composite1 = Image.composite(video1_frame, scene2_frame, mask1[f])
            composite2 = Image.composite(video2_frame, scene1_frame, mask2[f])

            mix1.append(np.asarray(composite1))
            mix2.append(np.asarray(composite2))

        output_path1 = output_path / file1
        output_path2 = output_path / file2

        output_path1.parent.mkdir(parents=True, exist_ok=True)
        output_path2.parent.mkdir(parents=True, exist_ok=True)

        ImageSequenceClip(mix1, fps=video1.fps).write_videofile(
            str(output_path1), audio=False, logger=None
        )

        ImageSequenceClip(mix2, fps=video2.fps).write_videofile(
            str(output_path2), audio=False, logger=None
        )


if __name__ == "__main__":
    main()
