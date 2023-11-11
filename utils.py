from pathlib import Path


def assert_dir(path, name: str):
    path = pathify(path)

    assert path.exists(), f"{name} not found."
    assert path.is_dir(), f"{name} must be a directory."


def assert_file(path, name: str, ext: str = None):
    path = pathify(path)

    assert path.exists(), f"{name} not found."
    assert path.is_file(), f"{name} must be a file."

    if ext is not None:
        ext = correct_suffix(ext)

        assert path.suffix == ext, f"{name} must be in a {ext} format."


def count_files(path, recursive=True, ext: str = None):
    path = pathify(path)
    pattern = "**/*" if recursive else "*"

    if ext is not None:
        pattern += correct_suffix(ext)

    return sum(1 for f in path.glob(pattern))


def correct_suffix(suffix: str) -> str:
    return suffix if suffix.startswith(".") else "." + suffix


def pathify(path):
    # TODO: assert path is str or Path
    return Path(path) if type(path) is str else path
