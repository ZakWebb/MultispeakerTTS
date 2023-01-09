#from tasks.vocoders import hifigan
#from tasks.vocoders import fastdiff
from tasks.vocoders.base_vocoder import get_vocoder_cls

_VOCODERS = {}

def get_vocoder(config):
    if config.get("vocoder") is None:
        raise ValueError("No defined vocoder in config file.")
    name = config["vocoder"]
    if _VOCODERS.get(name) is not None:
        return _VOCODERS[name]
    raise ValueError("{name} is not a defined text to phoneme operation.")