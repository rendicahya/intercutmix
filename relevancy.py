"""
This script generates the relevant object ID's and names based on the similarity of class names (actions) in the dataset and the object names covered in the UniDet object detector.
The results are put in JSON format in relevancy/unidet-matrix. This should be the first script to run before any other Python scripts.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from utils.config import Config
from utils.file_utils import *


def encode_cached(phrase, model):
    if phrase not in embed_bank:
        embedding = model.encode(phrase)
        embed_bank[phrase] = embedding

    return embed_bank[phrase]


def calc_similarity(phrase1, phrase2, model):
    emb1 = encode_cached(phrase1, model)
    emb2 = encode_cached(phrase2, model)

    return float(util.cos_sim(emb1, emb2))


conf = Config("config.json")
assert_file("config.json", ".json")

detector = conf.relevancy.detector.name
dataset_path = Path(conf.ucf101.path)
classnames_path = Path(conf.relevancy.detector.classnames)

assert_dir(dataset_path)
assert_file(classnames_path, ".json")

camelcase_tokenizer = re.compile(r"(?<!^)(?=[A-Z])")
n_subdir = sum([1 for f in dataset_path.iterdir() if f.is_dir()])
actions = [action.name for action in dataset_path.iterdir()]
avail_methods = conf.relevancy.avail_methods
embed_bank = {}

with open(classnames_path, "r") as file:
    classnames = json.load(file)

for model_name in avail_methods:
    print("Running model:", model_name)

    model = SentenceTransformer(model_name)
    embed_bank = {}
    df_data = []

    for subdir in tqdm(dataset_path.iterdir(), total=n_subdir):
        action = camelcase_tokenizer.sub(" ", subdir.name)
        row = [
            calc_similarity(action.lower(), obj.lower(), model) for obj in classnames
        ]

        df_data.append(row)

    df = pd.DataFrame(df_data, columns=classnames, index=actions)
    ids_output_dir = Path(f"relevancy/{detector}-ids") / model_name
    names_output_dir = Path(f"relevancy/{detector}-names") / model_name

    names_output_dir.mkdir(parents=True, exist_ok=True)
    ids_output_dir.mkdir(parents=True, exist_ok=True)

    for n in range(1, 6):
        sorted_ids = {}
        sorted_names = {}

        for i, row in enumerate(df.itertuples(index=False)):
            action = df.index[i]
            top_ids = np.argsort(row)[::-1][:n]
            top_names = df.columns[top_ids].to_list()

            sorted_ids.update({action: top_ids.tolist()})
            sorted_names.update({action: top_names})

        with open(ids_output_dir / f"top-{n}.json", "w") as f:
            json.dump(sorted_ids, f, indent=2)

        with open(names_output_dir / f"top-{n}.json", "w") as f:
            json.dump(sorted_names, f, indent=2)

    for thres in [i * 0.1 for i in range(1, 10)]:
        filtered_names = {}
        filtered_ids = {}

        for i, row in enumerate(df.itertuples(index=False)):
            action = df.index[i]
            ids_above = [i for i, val in enumerate(row) if val > thres]
            names_above = [col for col, val in zip(df.columns, row) if val > thres]

            filtered_names.update({action: names_above})
            filtered_ids.update({action: ids_above})

        with open(ids_output_dir / f"{thres:.1}.json", "w") as f:
            json.dump(filtered_ids, f)

        with open(names_output_dir / f"{thres:.1}.json", "w") as f:
            json.dump(filtered_names, f, indent=2)
