import importlib
import os
import subprocess
import shlex

from tasks.base_task import BaseTask
from tasks.text_cleaners import BaseTextCleaner, get_cleaner
from tasks.text_to_phoneme import BaseText2Phoneme, get_t2p
from data_gen.audio import AudioReader


_PREPROCESSORS = {}

def register_preprocessor(cls):
    _PREPROCESSORS[cls.__name__.lower()] = cls
    _PREPROCESSORS[cls.__name__] = cls
    return cls


def get_preprocessor_cls(cls):
    if cls in _PREPROCESSORS:
        return _PREPROCESSORS[cls]
    else:
        preprocessor_cls = cls
        pkg = ".".join(preprocessor_cls.split(".")[:-1])
        cls_name =preprocessor_cls.split(".")[-1]
        preprocessor_cls = getattr(importlib.import_module(pkg), cls_name)
        return preprocessor_cls


_NEEDED_DATA = {"raw_text", "data", "textgrid", "phonemes", "mels"}

class BasePreprocessTask(BaseTask):
    def __init__(self, config):
        super(BasePreprocessTask, self).__init__()
    
        self.raw_data_dir = config["raw_data_dir"]
        self.data_dir = config["data_dir"]
        self.sample_rate = config["sample_rate"]
        #self.meta_data = self.get_meta_data(config)

        self.train_percentage = config["train_percentage"]
        if self.train_percentage < 0 or self.train_percentage > 1:
            raise ValueError("Train Percentage should be between 0 and 1.")
        self.valid_percentage = config["valid_percentage"]
        if self.valid_percentage < 0 or self.valid_percentage > 1:
            raise ValueError("Valid Percentage should be between 0 and 1.")
        if self.train_percentage + self.valid_percentage >= 1:
            raise ValueError("Train Percentage and Valid Percentage should sum to less than 1.")
        
        self.audio = AudioReader(config)

        self.cleaner = get_cleaner(config)(config)
        self.t2p = get_t2p(config)(config)

        self.silence_boost = config.get("silence_boost", 1)
        self.mfa_processes = config.get("mfa_processes", 10)

    
    def build_dirs(self):
        os.makedirs(self.data_dir, exist_ok = True)
        for split in {"train", "valid", "test"}:
            os.makedirs(os.path.join(self.data_dir, split), exist_ok = True)
            for type in _NEEDED_DATA:
                os.makedirs(os.path.join(self.data_dir, split, type), exist_ok = True)

    def build_files(self):
        raise NotImplementedError

    def run_aligner(self):
        for split in {"train", "valid", "test"}:
            print("Aligning {} data", split)
            process = subprocess.Popen(shlex.split("mfa align {} english_us_arpa english_us_arpa {} -boost_silence={} -j={}".format(
                                                                os.path.join(self.data_dir, split, "data"),
                                                                os.path.join(self.data_dir, split, "textgrids"),
                                                                self.silence_boost,
                                                                self.mfa_processes
                                        )),
                                        stout=subprocess.PIPE,
                                        universal_newlines=True)
            while True:
                output = process.stdout.readline()
                print(output.strip())
                return_code = process.poll()
                if return_code is not None:
                    print('RETURN CODE', return_code)
                    for output in process.stdout.readlines():
                        print(output.strip())
                    break

    def process_textgrids(self):
        raise NotImplementedError
    
    def start(self):
        self.build_dirs()
        self.build_files()
        self.run_aligner()
        self.process_textgrids()