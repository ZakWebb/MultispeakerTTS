import json
import os
import torch
from torch.utils.data import Dataset

from data_gen.audio.audio_reader import AudioReader

_VALID_DATATYPES = {"raw_text", "cleaned_text", "phonemes", "mels", "wavs"}

class TTSDataset(Dataset):
    def __init__(self, config, input, output, train_type):
        super(TTSDataset).__init__()
        self.data_dir = os.path.join(config["data_dir"], train_type)
        self.train_type = train_type
        assert input in _VALID_DATATYPES
        self.input = input
        assert output in _VALID_DATATYPES
        self.output = output

        if "wavs" in {input, output}:
            self.audio = AudioReader(config)
            self.sample_rate = config["sample_rate"]
        
        self.get_files()

    
    def get_files(self):
        self.files = []
        with open(os.path.join(self.data_dir, self.train_type + "_metadata.csv")) as f:
            for line in f:
                self.files.append(line[:-1].split(",")[0]) # get rid of the trailing '\n' and take the first column
    
    def __len__(self):
        return len(self.files)
    
    def get_file_data(self, name, datatype):
        filename = os.path.join(self.data_dir, datatype, name)
        if datatype == "wavs":
            self.audio.load_wav(os.path.join(self.data_dir, datatype), name)
            data = self.audio.get_wav()
        elif datatype == "mels":
            filename = filename + ".mel"
            data = torch.load(filename)
        elif datatype == "phonemes":
            filename = filename + ".json"
            with open(filename, 'r') as f:
                data = json.load(f)
        else:
            filename = filename + ".txt"
            with open(filename, 'r') as f:
                data = f.read()
        
        return data

    def __getitem__(self, idx):
        input_item = self.get_file_data(self.files[idx], self.input)
        output_item = self.get_file_data(self.files[idx], self.output)

        return input_item, output_item