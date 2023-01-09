from tasks.base_conversion import BaseConversion
import importlib

_CLEANERS = {}


def register_cleaner(cls):
    _CLEANERS[cls.__name__.lower()] = cls
    _CLEANERS[cls.__name__] = cls
    return cls


def get_cleaner_cls(cls):
    if cls in _CLEANERS:
        return _CLEANERS[cls]
    else:
        cleaner_cls = cls
        pkg = ".".join(cleaner_cls.split(".")[:-1])
        cls_name =cleaner_cls.split(".")[-1]
        cleaner_cls = getattr(importlib.import_module(pkg), cls_name)
        return cleaner_cls

class BaseTextCleaner(BaseConversion):
    def convert(self, text):
        raise NotImplemented