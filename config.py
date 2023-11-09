import json
from pathlib import Path


class Dict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Config(object):
    @staticmethod
    def __load__(data):
        if type(data) is dict:
            return Config.load_dict(data)
        elif type(data) is list:
            return Config.load_list(data)
        else:
            return data

    @staticmethod
    def load_json(path: Path):
        assert path.exists(), "Config file not found."
        assert path.is_file(), "Config file must be a file."
        assert path.suffix == ".json", "Config file must be a JSON file."

        with open(path, "r") as f:
            result = Config.__load__(json.loads(f.read()))

        return result
