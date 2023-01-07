import importlib
import os
from ..base_task import BaseTask
from ..text_cleaners import BaseTextCleaner, get_cleaner_cls
from ...data_gen.audio import AudioReader


_PREPROCESSORS = {}

def register_preprocessor(cls):
    _PREPROCESSORS[cls.__name__.lower()] = cls
    _PREPROCESSORS[cls.__name__] = cls
    return cls


def get_preprocessor_cls(hparams):
    if hparams['preprocessor'] in _PREPROCESSORS:
        return _PREPROCESSORS[hparams['preprocessor']]
    else:
        preprocessor_cls = hparams['preprocessor']
        pkg = ".".join(preprocessor_cls.split(".")[:-1])
        cls_name =preprocessor_cls.split(".")[-1]
        preprocessor_cls = getattr(importlib.import_module(pkg), cls_name)
        return preprocessor_cls


_NEEDED_DATA = {"raw_text", "cleaned_text", "phonemes", "mels", "wavs"}

class BasePreprocessTask(BaseTask):
    def __init__(self, config):
        super(BasePreprocessTask).__init__(self)
        
        self.raw_data_folder = config["raw_data_folder"]
        self.data_folder = config["data_folder"]
        self.sample_rate = config["sample_rate"]
        #self.meta_data = self.get_meta_data(config)

        self.test_percentage = config["test_percentage"]
        if self.test_percentage < 0 or self.test_percentage > 1:
            raise ValueError("Test Percentage should be between 0 and 1.")
        self.valid_percentage = config["valid_percentage"]
        if self.valid_percentage < 0 or self.valid_percentage > 1:
            raise ValueError("Valid Percentage should be between 0 and 1.")
        if self.test_percentage + self.valid_percentage >= 1:
            raise ValueError("Test Percentage and Valid Percentage should sum to less than 1.")
        
        self.audio = AudioReader(config)

        self.cleaner = get_cleaner_cls(config["cleaner"])

    
    def build_folders(self):
        os.makedirs(self.data_folder, exists_ok = True)
        for split in {"train", "valid", "test"}:
            os.makedirs(os.join(self.data_folder, split),exists_ok = True)
            for type in _NEEDED_DATA:
                os.makedirs(os.join(self.data_folder, split, type))

    def build_files(self):
        raise NotImplemented
    
    def start(self):
        self.build_folders()
        self.build_files()