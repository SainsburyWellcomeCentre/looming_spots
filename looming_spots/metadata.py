import os
import configobj
from configobj import ConfigObj

DEFAULT_PATH = './metadata.cfg'


def load_from_config(key, path=None):
    if path is None:
        path = DEFAULT_PATH
    config = ConfigObj(path, unrepr=True)
    return config[key]


def save_key_to_config(config, key, value):
    config[key] = value
    config.write()


def load_config(path=None):
    if path is None:
        path = DEFAULT_PATH
    if os.path.isfile(path):
        return ConfigObj(path, indent_type='    ', unrepr=False)
    return ConfigObj(path, indent_type='    ', unrepr=False)


def initialise_config(config, key):
    if key not in config:
        config[key] = {}
    return config


def load_nest_level_metadata(metadata, key):  # FIXME:
    if key in metadata:
        return metadata
    for item in metadata:
        if item == key:
            print(item, key)
            return metadata
        if isinstance(metadata[item], configobj.Section):
            print('configobj section, repeating for {}'.format(metadata[item]))
            load_nest_level_metadata(metadata[item], key)
    return 'key not present'




