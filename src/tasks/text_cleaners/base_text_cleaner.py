from base_conversion import BaseConversion
import importlib

_CLEANERS = {}


def register_cleaner(cls):
    _CLEANERS[cls.__name__.lower()] = cls
    _CLEANERS[cls.__name__] = cls
    return cls


def get_cleaner_cls(hparams):
    if hparams['cleaner'] in _CLEANERS:
        return _CLEANERS[hparams['cleaner']]
    else:
        preprocessor_cls = hparams['cleaner']
        pkg = ".".join(preprocessor_cls.split(".")[:-1])
        cls_name =preprocessor_cls.split(".")[-1]
        preprocessor_cls = getattr(importlib.import_module(pkg), cls_name)
        return preprocessor_cls

class BaseTextCleaner(BaseConversion):
    def convert(self, text):
        raise NotImplemented