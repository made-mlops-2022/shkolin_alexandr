import yaml


def read_config(path: str):
    """Read config"""

    with open(path, 'r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)
