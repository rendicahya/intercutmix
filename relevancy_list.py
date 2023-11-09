import json
from pathlib import Path

import numpy as np
import pandas as pd

dataset_name = "unidet"
matrix_path = f"{dataset_name}-matrix"

assert matrix_path.exists(), "Matrix path not found."
assert matrix_path.is_dir(), "Matrix path must be a directory."

for csv in Path(matrix_path).iterdir():
    df = pd.read_csv(csv, index_col=0, engine="pyarrow").astype(float)
    names_output_dir = Path(f"{dataset_name}-relevant-names") / csv.stem
    ids_output_dir = Path(f"{dataset_name}-relevant-ids") / csv.stem

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
