from tasks.base_conversion import BaseConversion
from data_gen.symbols.symbols import symbols
import importlib

_T2PS = {}


def register_t2p(cls):
    _T2PS[cls.__name__.lower()] = cls
    _T2PS[cls.__name__] = cls
    return cls


def get_t2p_cls(hparams):
    if hparams['text_to_phoneme'] in _T2PS:
        return _T2PS[hparams['text_to_phoneme']]
    else:
        t2p_cls = hparams['text_to_phoneme']
        pkg = ".".join(t2p_cls.split(".")[:-1])
        cls_name =t2p_cls.split(".")[-1]
        t2p_cls = getattr(importlib.import_module(pkg), cls_name)
        return t2p_cls

class BaseText2Phoneme(BaseConversion):
    def __init__(self, config):
        super(BaseText2Phoneme, self).__init__(config)

        # We might need to change this for other languages
        self.symbols = symbols

    def convert(self, text):
        raise NotImplementedError