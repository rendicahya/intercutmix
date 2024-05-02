import cv2
import mmcv
import numpy as np


def cutmix(actor_path, scene_path, mask_bundle):
    if not actor_path.is_file() or not actor_path.exists():
        print("Not a file or not exists:", actor_path)
        return None

    if not scene_path.is_file() or not scene_path.exists():
        print("Not a file or not exists:", scene_path)
        return None

    actor_frames = mmcv.VideoReader(str(actor_path))
    w, h = actor_frames.resolution
    blank = np.zeros((h, w), np.uint8)
    scene_frame = None
    scene_info = mmcv.VideoReader(str(scene_path))
    scene_w, scene_h = scene_info.resolution

    for f, actor_frame in enumerate(actor_frames):
        if f == len(mask_bundle) - 1:
            return

        if scene_frame is None:
            scene_frames = mmcv.VideoReader(str(scene_path))
            scene_frame = scene_frames.read()

        if scene_w != w or scene_h != h:
            scene_frame = cv2.resize(scene_frame, (w, h))

        actor_mask = mask_bundle[f]

        if actor_mask is None:
            actor_mask = blank

        scene_mask = 255 - actor_mask

        actor = cv2.bitwise_and(actor_frame, actor_frame, mask=actor_mask)
        scene = cv2.bitwise_and(scene_frame, scene_frame, mask=scene_mask)

        mix = actor + scene
        scene_frame = scene_frames.read()

        yield cv2.cvtColor(mix, cv2.COLOR_BGR2RGB)
