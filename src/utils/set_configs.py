import argparse
import os
import yaml

def override_confige(old_config, new_config):
    for k, v in new_config.items():
        if isinstance(v, dict) and k in old_config:
            override_confige(old_config[k], new_config[k])
        else:
            old_config[k] = v

def set_config(print_hparams=False):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='../configs/config_base.yaml')


    args, unknown = parser.parse_known_args()

    config_file_name = args.config

    configs = temp_configs = load_config(config_file_name)
    while temp_configs.get("base_config") is not None:
        config_file_name = os.path.join(os.path.dirname(config_file_name), temp_configs["base_config"])
        temp_configs = load_config(config_file_name)
        override_confige(configs, temp_configs)
    return configs

def load_config(fname):
    if not os.path.exists(fname):
        return {}
    with open(fname) as f:
        _config = yaml.safe_load(f)
    return _config