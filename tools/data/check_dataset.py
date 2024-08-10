import sys

sys.path.append(".")

from pathlib import Path

from config import settings as C
from python_file import count_files

root = Path.cwd() / "data"
seed = C.active.random_seed
multiplication = C.cutmix.multiplication


def check_count(path, expected):
    count = count_files(path) if path.exists() else "Not exist"

    print(
        path.relative_to(root),
        f"[{count}/{expected}]",
        "OK" if count == expected else "Error",
    )


def check(mask, mix, n_videos, multiplication):
    expected = n_videos + 1
    count = count_files(mask) if mask.exists() else "Not exist"

    print(
        mask.relative_to(root),
        f"[{count}/{expected}]",
        "OK" if count == expected else "!!!",
    )

    expected = n_videos * multiplication + 1
    count = count_files(mix) if mask.exists() else "Not exist"

    print(
        mix.relative_to(root),
        f"[{count}/{expected}]",
        "OK" if count == expected else "!!!",
    )


for dataset in "ucf101", "hmdb51":
    n_videos = C[dataset].n_videos

    for detector in ("UniDet",):
        # mask = root / dataset / detector / "detect/mask"
        # mix = root / dataset / detector / f"detect/mix-{seed}"
        # check(mask, mix, n_videos, multiplication)

        # mask = root / dataset / detector / "detect/REPP/mask"
        # mix = root / dataset / detector / f"detect/REPP/mix-{seed}"
        # check(mask, mix, n_videos, multiplication)

        # mask = root / dataset / detector / "select/actorcutmix/mask"
        # mix = root / dataset / detector / f"select/actorcutmix/mix-{seed}"
        # check(mask, mix, n_videos, multiplication)

        intercutmix = root / dataset / detector / "select/intercutmix"

        for relev_method in (intercutmix / "mask").iterdir():
            for relev in relev_method.iterdir():
                check_count(relev, n_videos + 1)

        for relev_method in (intercutmix / f"mix-{seed}").iterdir():
            for relev in relev_method.iterdir():
                check_count(relev, n_videos * multiplication + 1)
