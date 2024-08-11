import sys

sys.path.append(".")

from pathlib import Path

from config import settings as C
from python_file import count_files

root = Path.cwd() / "data"
seed = C.active.random_seed
multiplication = C.cutmix.multiplication


def check(path, expected):
    count = count_files(path) if path.exists() else "Not exist"

    print(
        path.relative_to(root),
        f"[{count}/{expected}]",
        "OK" if count == expected else "Error",
    )


for dataset in "ucf101", "hmdb51":
    n_videos = C[dataset].n_videos

    for detector in ("UniDet",):
        mask = root / dataset / detector / "detect/mask"
        mix = root / dataset / detector / f"detect/mix-{seed}"
        check(mask, n_videos + 1)
        check(mix, n_videos * multiplication + 1)

        mask = root / dataset / detector / "detect/REPP/mask"
        mix = root / dataset / detector / f"detect/REPP/mix-{seed}"
        check(mask, n_videos + 1)
        check(mix, n_videos * multiplication + 1)

        mask = root / dataset / detector / "select/actorcutmix/mask"
        mix = root / dataset / detector / f"select/actorcutmix/mix-{seed}"
        check(mask, n_videos + 1)
        check(mix, n_videos * multiplication + 1)

        mask = root / dataset / detector / "select/actorcutmix/REPP/mask"
        mix = root / dataset / detector / f"select/actorcutmix/REPP/mix-{seed}"
        check(mask, n_videos + 1)
        check(mix, n_videos * multiplication + 1)

        intercutmix = root / dataset / detector / "select/intercutmix"

        for relev_method in (intercutmix / "mask").iterdir():
            for relev in relev_method.iterdir():
                check(relev, n_videos + 1)

        for relev_method in (intercutmix / f"mix-{seed}").iterdir():
            for relev in relev_method.iterdir():
                check(relev, n_videos * multiplication + 1)

        for relev_method in (intercutmix / "REPP/mask").iterdir():
            for relev in relev_method.iterdir():
                check(relev, n_videos + 1)

        for relev_method in (intercutmix / f"REPP/mix-{seed}").iterdir():
            for relev in relev_method.iterdir():
                check(relev, n_videos * multiplication + 1)
