import json
from pathlib import Path

'''
Usage:

from config import Config
conf = Config('config.json')
print(conf.some.value)
'''


class Dict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Config(object):
    def __new__(cls, path: str):
        path = Path(path)

        assert path.exists(), "Config file not found."
        assert path.is_file(), "Config file must be a file."
        assert path.suffix == ".json", "Config file must be a JSON file."

        with open(path, "r") as f:
            result = Config.__load__(json.loads(f.read()))

        return result

    @staticmethod
    def __load__(data):
        if type(data) is dict:
            return Config.load_dict(data)
        elif type(data) is list:
            return Config.load_list(data)
        else:
            return data

    @staticmethod
    def load_dict(data: dict):
        result = Dict()
        for key, value in data.items():
            result[key] = Config.__load__(value)

        return result

    @staticmethod
    def load_list(data: list):
        result = [Config.__load__(item) for item in data]

        return result
