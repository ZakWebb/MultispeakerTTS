
from tasks.base_task import BaseTask
import importlib

_P2M = {}

def register_p2m(cls):
    _P2M[cls.__name__.lower()] = cls
    _P2M[cls.__name__] = cls
    return cls


def get_p2m_cls(cls):
    if cls in _P2M:
        return _P2M[cls]
    else:
        preprocessor_cls = cls
        pkg = ".".join(preprocessor_cls.split(".")[:-1])
        cls_name =preprocessor_cls.split(".")[-1]
        preprocessor_cls = getattr(importlib.import_module(pkg), cls_name)
        return preprocessor_cls


class BasePhoneme2Mel(BaseTask):
    def start(self, phoneme):
        raise NotImplemented