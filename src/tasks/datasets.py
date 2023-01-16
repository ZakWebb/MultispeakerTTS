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
    post_outputs = collater(output, pre_outputs)

    return post_inputs, post_outputs

def collater(datatype, data):
    cur_data=[]
    if datatype == "mels":
        for point in data:
            cur_data.append(torch.transpose(point,0,1))
    else:
        cur_data = data
            
    collated = pad_sequence(cur_data, batch_first=True)

    if datatype == "mels":
        collated = torch.transpose(collated, 1, 2)

    return collated


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
        filename = os.path.join(self.data_dir, datatype, name)
        if datatype == "wavs":
            self.audio.load_wav(os.path.join(self.data_dir, datatype), name)
            data = self.audio.get_wav()
            data = torch.from_numpy(data)
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

        return {"input": input_item, "output": output_item}

def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded