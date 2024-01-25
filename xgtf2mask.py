from pathlib import Path
from typing import Union

import cv2
import numpy as np
from assertpy.assertpy import assert_that
from bs4 import BeautifulSoup
from python_config import Config
from python_file import count_files
from python_video import video_info
from tqdm import tqdm


def parse_xgtf(path: Union[Path, str], action_only: bool = False):
    assert_that(path).is_file().is_readable()

    with open(path) as f:
        try:
            soup = BeautifulSoup(f, "xml")
        except:
            print("Error reading", path)
            return None

    all_bbox = {}
    data = soup.find("data")

    if data is None:
        return None

    for sourcefile in data.find_all("sourcefile"):
        for person in sourcefile.find_all("object", {"name": "PERSON"}):
            person_id = int(person["id"])

            if person_id in all_bbox:
                continue

            person_locations = person.find("attribute", {"name": "Location"})
            person_bbox = {}
            person_action = person.find("data:bvalue", {"value": "true"})

            if not person_action:
                continue

            act_start, act_end = [int(i) for i in person_action["framespan"].split(":")]

            for bbox in person_locations.find_all("data:bbox"):
                start, end = [int(i) for i in bbox["framespan"].split(":")]

                if action_only:
                    if act_start <= start <= act_end or act_start <= end <= act_end:
                        start = max(start, act_start)
                        end = min(end, act_end)

                        for frame in range(start - 1, end):
                            bbox_data = {
                                frame: (
                                    int(bbox["x"]),
                                    int(bbox["y"]),
                                    int(bbox["width"]),
                                    int(bbox["height"]),
                                )
                            }

                            person_bbox.update(bbox_data)
                else:
                    for frame in range(start - 1, end):
                        bbox_data = {
                            frame: (
                                int(bbox["x"]),
                                int(bbox["y"]),
                                int(bbox["width"]),
                                int(bbox["height"]),
                            )
                        }

                        person_bbox.update(bbox_data)

            all_bbox.update({person_id: person_bbox})

    return all_bbox


def main():
    conf = Config("config.json")
    xgtf_dir = Path(conf.xgtf.path)
    ucf101_dir = Path(conf.ucf101.path)
    ucf101_ext = conf.ucf101.ext
    mask_ext = conf.xgtf.mask.ext
    output_root = Path(conf.xgtf.mask.path)
    bar = tqdm(total=count_files(xgtf_dir))

    assert_that(xgtf_dir).is_directory().is_readable()
    assert_that(ucf101_dir).is_directory().is_readable()

    for action in xgtf_dir.iterdir():
        for xgtf in action.iterdir():
            bar.set_description(xgtf.name)

            if xgtf.suffix != ".xgtf":
                continue

            people_bbox = parse_xgtf(xgtf, action_only=conf.xgtf.mask.action_only)

            if not people_bbox:
                continue

            video_path = ucf101_dir / action.name / (xgtf.with_suffix(ucf101_ext).name)
            output_dir = output_root / action.name / xgtf.stem

            output_dir.mkdir(parents=True, exist_ok=True)

            info = video_info(video_path)
            n_frames = info["n_frames"]
            width, height = info["width"], info["height"]

            for i in range(n_frames):
                mask = np.zeros((height, width), dtype=np.uint8)
                output_file = output_dir / (f"%05d{mask_ext}" % i)

                for person_id, person_bbox in people_bbox.items():
                    if i not in person_bbox:
                        continue

                    x1, y1, w, h = person_bbox[i]
                    x2 = x1 + w
                    y2 = y1 + h
                    mask[y1:y2, x1:x2] = 255

                cv2.imwrite(str(output_file), mask)

            bar.update(1)

    bar.close()


if __name__ == "__main__":
    main()
