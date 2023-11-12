import json
import random
from pathlib import Path

import cv2
import numpy as np
from config import Config
from tqdm import tqdm
from utils.utils import *

conf = Config("config.json")
dataset_path = Path(conf.ucf101.path)
unidet_json_path = Path(conf.unidet.input.json)
relevant_object_json = Path(conf.relevancy.json)
confidence_thres = conf.unidet.confidence
unified_label = Path("learned_mAP.json")
output_video_path = Path(conf.unidet.output.video.path)
output_mask_dir = Path(conf.unidet.output.mask.path)

assert_file("config.json", "Configuration")
assert_dir(dataset_path, "Dataset path")
assert_dir(unidet_json_path, "UniDet JSON path")
assert_file(relevant_object_json, "Relevant object JSON", ".json")
assert_file(unified_label, "Unified label", ".json")

n_files = count_files(dataset_path, ext=conf.ucf101.ext)

with open(unified_label, "r") as f:
    unified_label_file = json.load(f)

thing_classes = [
    "{}".format([xx for xx in x["name"].split("_") if xx != ""][0])
    for x in unified_label_file["categories"]
]

with open(relevant_object_json, "r") as f:
    relevant_ids = json.load(f)

colors = [
    (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
    for _ in range(len(thing_classes))
]

common_obj = "Person", "Man", "Woman"
common_ids = [thing_classes.index(i) for i in common_obj]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

with tqdm(total=n_files) as bar:
    for action in dataset_path.iterdir():
        relevant_obj = [*relevant_ids[action.name], *common_ids]

        for file in action.iterdir():
            bar.set_description(file.name)

            input_video = cv2.VideoCapture(str(file))
            width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(input_video.get(cv2.CAP_PROP_FPS))

            if conf.unidet.output.video.generate:
                output_video = (
                    output_video_path / action.name / file.with_suffix(".mp4").name
                )

                output_video.parent.mkdir(parents=True, exist_ok=True)

                video_writer = cv2.VideoWriter(
                    str(output_video),
                    fourcc,
                    fps,
                    (width, height),
                )

            json_file = unidet_json_path / action.name / file.with_suffix(".json").name

            if not json_file.exists():
                print("JSON file not found:", json_file.name)
                continue

            with open(json_file, "r") as f:
                box_data = json.load(f)

            i = 0

            while input_video.isOpened():
                ret, frame = input_video.read()

                if not ret:
                    break

                if str(i) not in box_data.keys():
                    continue

                output_mask_path = (
                    output_mask_dir / action.name / file.name / f"{i:05}.png"
                )
                output_mask = np.zeros(frame.shape)

                output_mask_path.parent.mkdir(exist_ok=True, parents=True)

                for box, confidence, class_id in box_data[str(i)]:
                    if confidence < confidence_thres or class_id not in relevant_obj:
                        continue

                    x1, y1, x2, y2 = [round(i) for i in box]
                    output_mask[y1:y2, x1:x2] = 255

                    cv2.imwrite(str(output_mask_path), output_mask)

                    if conf.unidet.output.video.generate:
                        text = f"{thing_classes[class_id]} {confidence:.02}"
                        font = cv2.FONT_HERSHEY_PLAIN
                        font_size = 1.2
                        font_weight = 1
                        text_size = cv2.getTextSize(text, font, font_size, font_weight)[
                            0
                        ]
                        text_width, text_height = text_size[:2]
                        text_x = x1
                        text_y = y1
                        box_thickness = 2

                        cv2.rectangle(
                            frame, (x1, y1), (x2, y2), colors[class_id], box_thickness
                        )
                        cv2.rectangle(
                            frame,
                            (text_x - 1, text_y - int(text_height * 2)),
                            (text_x + int(text_width * 1.1), text_y),
                            colors[class_id],
                            cv2.FILLED,
                        )
                        cv2.putText(
                            frame,
                            text,
                            (x1 + 3, y1 - 5),
                            font,
                            font_size,
                            (255, 255, 255),
                            font_weight,
                        )

                if conf.unidet.output.video.generate:
                    video_writer.write(frame)

                i += 1

            if conf.unidet.output.video.generate:
                video_writer.release()

            input_video.release()
            bar.update(1)
