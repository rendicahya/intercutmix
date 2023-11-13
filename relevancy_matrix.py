"""
This script generates the relevancy matrix between the class names (actions) and the object names covered in the UniDet object detector.
The resulting matrices are put in relevancy/unidet-matrix. This should be the first script to run before any other Python scripts.
"""

import json
import re
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from utils.config import Config
from utils.file_utils import *


def encode_cached(phrase, model, embed_bank):
    if phrase not in embed_bank:
        embedding = model.encode(phrase)
        embed_bank[phrase] = embedding

    return embed_bank[phrase]


def calc_similarity(phrase1, phrase2, model, embed_bank):
    emb1 = encode_cached(phrase1, model, embed_bank)
    emb2 = encode_cached(phrase2, model, embed_bank)

    return float(util.cos_sim(emb1, emb2))


conf = Config("config.json")
assert_file('config.json', 'Configuration','.json')

detector = "unidet"
dataset_path = Path(conf.ucf101.path)
classnames_path = Path(f"relevancy/{detector}-classnames.json")

assert_file("config.json", "Configuration", ".json")
assert_dir(dataset_path, "Dataset path")
assert_file(classnames_path, "Classname", ".json")

camelcase_tokenizer = re.compile(r"(?<!^)(?=[A-Z])")
n_subdir = sum([1 for f in dataset_path.iterdir() if f.is_dir()])
actions = [action.name for action in dataset_path.iterdir()]
output_dir = Path(f"relevancy/{detector}-matrix")
avail_methods = conf.relevancy.avail_methods

output_dir.mkdir(exist_ok=True)

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
            calc_similarity(action.lower(), obj.lower(), model, embed_bank)
            for obj in classnames
        ]

        df_data.append(row)

    pd.DataFrame(df_data, columns=classnames, index=actions).to_csv(
        output_dir / f"{model_name}.csv"
    )
