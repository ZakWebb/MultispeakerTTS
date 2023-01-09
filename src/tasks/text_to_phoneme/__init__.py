from tasks.text_to_phoneme.base_text_to_phoneme import get_t2p_cls, BaseText2Phoneme
from tasks.text_to_phoneme.simple_english_t2p import SimpleEnglishT2P

_T2P = {"Simple English": SimpleEnglishT2P}

def get_t2p(config):
    if config.get("text_to_phoneme") is None:
        raise ValueError("No defined text to phoneme in config file.")
    name = config["text_to_phoneme"]
    if _T2P.get(name) is not None:
        return _T2P[name]
    raise ValueError("{name} is not a defined text to phoneme operation.")