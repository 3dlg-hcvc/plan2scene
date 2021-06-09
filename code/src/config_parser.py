import json
import os.path as osp
import logging


class Config:
    """
    Configuration item.
    """

    def __init__(self, config_dict: dict):
        """
        Initialize configuration item using a dictionary.
        :param d:
        """
        for k, v in config_dict.items():
            if isinstance(v, dict):
                v = Config(v)
            self.__dict__[k] = v

    def __getitem__(self, key):
        return self.__dict__[key]


def parse_config(config_path: str):
    """
    Parses a json config file into a Config object.
    :param config_path: Path to the json config file.
    """
    if not osp.exists(config_path):
        logging.warning(f"Config file not found: {config_path}")
        return None

    with open(config_path, "r") as f:
        config_dict = json.loads(f.read())
    if isinstance(config_dict, dict):
        return Config(config_dict)
    else:
        return config_dict
