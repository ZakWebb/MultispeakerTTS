from tasks.phoneme_to_mel.base_phoneme_to_mel import BasePhoneme2Mel, get_p2m_cls
_P2M = {}

def get_p2m(config):
    if config.get("phoneme to mel") is None:
        raise ValueError("No defined phoneme to mel operation in config file.")
    name = config["phoneme_to_mel"]
    if _P2M.get(name) is not None:
        return _P2M[name]
    raise ValueError("{name} is not a defined phoneme to mel operation.")