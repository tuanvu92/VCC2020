import json


def load_config(json_config_path: str) -> dict:
    with open(json_config_path, 'r') as f:
        configs_raw_string = f.read()
        configs = json.loads(configs_raw_string)
        return configs
