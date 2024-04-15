import random
from pathlib import Path

from config import settings as conf

hmdb51_dir = Path(conf.hmdb51.path)
raw_split_dir = hmdb51_dir.parent / "testTrainMulti_7030_splits"
split_no = conf.hmdb51.split
action_list = []
train_list = []
test_list = []

random.seed(conf.active.random_seed)

for file in raw_split_dir.iterdir():
    ending = f"_test_split{split_no}.txt"
    action = file.name.split(ending)[0]

    if not file.is_file() or not file.name.endswith(ending):
        continue

    if not action in action_list:
        action_list.append(action)

    action_index = action_list.index(action)

    with open(file) as f:
        for line in f:
            filename, category = line.strip().split()
            new_line = f"{action}/{filename} {action_index}"

            if int(category) == 1:
                train_list.append(new_line)
            elif int(category) == 2:
                test_list.append(new_line)

random.shuffle(train_list)
random.shuffle(test_list)

with open(hmdb51_dir.parent / "train.txt", "w") as f:
    f.write("\n".join(train_list))

with open(hmdb51_dir.parent / "test.txt", "w") as f:
    f.write("\n".join(test_list))
