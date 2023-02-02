import functools
import json
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from data_gen.audio.audio_reader import AudioReader

_VALID_DATATYPES = {"raw_text", "cleaned_text", "phonemes", "mels", "wavs"}

def get_TTSDataset_collater(input, output):
    if input not in _VALID_DATATYPES:
        raise ValueError("{in} not a valid datatype".format(input))
    if output not in _VALID_DATATYPES:
        raise ValueError("{out} not a valid datatype".format(output))
    
    return functools.partial(TTSDataset_collater, input, output)

def TTSDataset_collater(input, output, data):
    pre_inputs = [sample["input"] for sample in data]
    pre_outputs = [sample["output"] for sample in data]

    post_inputs = collater(input, pre_inputs)
    post_outputs  = collater(output, pre_outputs)

    ret = {"inputs": post_inputs,
            "outputs": post_outputs}

    return ret

def collater(datatype, data):
    cur_data=[]
    lens = []
    if datatype in {"mels", "wavs"}:
        for point in data:
            cur_data.append(torch.transpose(point,0,1))
    else:
        cur_data = data

    for point in cur_data:
        lens.append(point.size(0))

    lens = torch.tensor(lens)
            
    collated = pad_sequence(cur_data, batch_first=True)
    mask = get_mask_from_lens(lens, collated.size())

    if datatype in  {"mels", "wavs"}:
        collated = torch.transpose(collated, 1, 2)
        mask = torch.transpose(mask, 1, 2)
    
    ret = {"data": collated,
            "mask": mask,
            "lens": lens}

    return ret

def get_mask_from_lens(lens, req_size):
    batch_size = lens.shape[0]
    max_len = torch.max(lens).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1)
    mask = (ids >= lens.unsqueeze(1).expand(-1, max_len))

    while (len(mask.size())) < len(req_size):
        mask = mask.unsqueeze(-1)
    
    mask = mask.expand(req_size)

    return mask.clone()



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
        
        if "mels" in {input, output}:
            self.n_mel_channels = config["n_mel_channels"]
        
        self.get_files()

    
    def get_files(self):
        self.files = []
        with open(os.path.join(self.data_dir, self.train_type + "_metadata.csv")) as f:
            for line in f:
                self.files.append(line[:-1].split(",")[0]) # get rid of the trailing '\n' and take the first column
    
    def __len__(self):
        return len(self.files)
    
    def get_file_data(self, name, datatype):
        filename = self.data_dir
        if datatype == "wavs":
            self.audio.load_wav(os.path.join(self.data_dir, "data"), name)
            data = self.audio.get_wav()
            data = torch.from_numpy(data).unsqueeze(0)
        elif datatype == "mels":
            filename = os.path.join(filename, "mels", name + ".mel")
            data = torch.load(filename)
        elif datatype == "phonemes":
            filename = os.path.join(filename, "phonemes", name + ".json")
            with open(filename, 'r') as f:
                data = json.load(f)
        elif datatype == "cleaned_text":
            filename = os.path.join(filename, "data", name + ".txt")
            with open(filename, 'r') as f:
                data = f.read()
        else:
            filename = os.path.join(filename, datatype, name + ".txt")
            with open(filename, 'r') as f:
                data = f.read()
        
        return data

    def __getitem__(self, idx):
        input_item = self.get_file_data(self.files[idx], self.input)
        output_item = self.get_file_data(self.files[idx], self.output)

        return {"input": input_item, "output": output_item}