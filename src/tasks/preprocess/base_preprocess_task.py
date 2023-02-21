import importlib
import os
import subprocess
import shlex

from tasks.base_task import BaseTask
from tasks.text_cleaners import BaseTextCleaner, get_cleaner
from tasks.text_to_phoneme import BaseText2Phoneme, get_t2p
from data_gen.audio import AudioReader
import numpy as np


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


_NEEDED_DATA = {"raw_text", "data", "textgrids", "phonemes", "mels", "durations", "energy", "f0"}

class BasePreprocessTask(BaseTask):
    def __init__(self, config):
        super(BasePreprocessTask, self).__init__()
    
        self.raw_data_dir = config["raw_data_dir"]
        self.data_dir = config["data_dir"]
        self.sample_rate = config["sample_rate"]
        self.hop_length = config["hop_length"]
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
        self.run_align = config.get("run_align", True)

    
    def build_dirs(self):
        os.makedirs(self.data_dir, exist_ok = True)
        for split in {"train", "valid", "test"}:
            os.makedirs(os.path.join(self.data_dir, split), exist_ok = True)
            for type in _NEEDED_DATA:
                os.makedirs(os.path.join(self.data_dir, split, type), exist_ok = True)

    def build_files(self):
        raise NotImplementedError

    def check_dir_complete(self, split, datatype):
        raise NotImplementedError

    def aligner(self):
        for split in {"train", "valid", "test"}:
            print("Aligning {} data".format(split))
            all_correct = self.check_dir_complete(split, "textgrids")
            # process = subprocess.Popen(shlex.split("mfa validate \"{}\" english_us_arpa english_us_arpa".format(
            #                                                     os.path.join(self.data_dir, split, "data")
            #                             )),
            #                             stdout=subprocess.PIPE,
            #                             universal_newlines=True)
            # while True:
            #     output = process.stdout.readline()
            #     print(output.strip())
            #     return_code = process.poll()
            #     if return_code is not None:
            #         print('RETURN CODE', return_code)
            #         for output in process.stdout.readlines():
            #             print(output.strip())
            #         break
            if not all_correct:
                process = subprocess.Popen(shlex.split("mfa align \"{}\" english_us_arpa english_us_arpa \"{}\" -boost_silence={} -j={} --clean".format(
                                                                    os.path.join(self.data_dir, split, "data"),
                                                                    os.path.join(self.data_dir, split, "textgrids"),
                                                                    self.silence_boost,
                                                                    self.mfa_processes
                                            )),
                                            stdout=subprocess.PIPE,
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
        if self.run_align:
            self.aligner()
        self.process_textgrids()

    def average_by_duration(self, x, durs):
        length = sum(durs)
        durs_cum = np.cumsum(np.pad(durs, (1, 0), mode='constant'))

        # calculate charactor f0/energy
        if len(x.shape) == 2:
            x_char = np.zeros((durs.shape[0], x.shape[1]), dtype=np.float32)
        else:
            x_char = np.zeros((durs.shape[0],), dtype=np.float32)
        for idx, start, end in zip(range(length), durs_cum[:-1], durs_cum[1:]):
            values = x[start:end][np.where(x[start:end] != 0.0)[0]]
            x_char[idx] = np.mean(values, axis=0) if len(values) > 0 else 0.0  # np.mean([]) = nan.

        return x_char.astype(np.float32)