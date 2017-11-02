from configobj import ConfigObj

DEFAULT_PATH = './metadata.cfg'


def load_from_config(key, path=None):
    if path is None:
        path = DEFAULT_PATH
    config = ConfigObj(path)
    return config[key]


def save_key_to_config(key, value, path=None):
    if path is None:
        path = DEFAULT_PATH
    config = ConfigObj(path, indent_type='    ')
    config[key] = value
    config.write()


