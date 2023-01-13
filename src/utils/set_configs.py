import argparse
import os
import yaml

def override_configs(old_config, new_config):
    for k, v in new_config.items():
        if isinstance(v, dict) and k in old_config:
            override_configs(old_config[k], new_config[k])
        else:
            old_config[k] = v

def set_config():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='../configs/config_base.yaml')


    args, unknown = parser.parse_known_args()

    config_file_name = args.config

    config = recursive_load(config_file_name)

    return config

def recursive_load(config_file_name):
    base_file = config_file_name
    configs = {}
    while base_file is not None:
        temp_configs = load_config(base_file)
        if temp_configs.get("model_config") is not None:
            temp_configs[temp_configs["task"] + "_config"] = recursive_load(os.path.join(os.path.dirname(base_file), temp_configs["model_config"]))
        if temp_configs.get("base_config") is not None:
            base_file = os.path.join(os.path.dirname(base_file), temp_configs["base_config"])
        else:
            base_file = None
        override_configs(configs, temp_configs)
    
    return configs

def load_config(fname):
    if not os.path.exists(fname):
        return {}
    with open(fname) as f:
        _config = yaml.safe_load(f)
    return _config