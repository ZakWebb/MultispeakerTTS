import os
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import pandas as pd
import numpy as np
import librosa
from functools import partial

import lightning as L

import torch
from torch.utils.data import random_split, DataLoader, Dataset


from utils import prepare_mels

VALID_DATA_TYPES_LJSPEECH = ["mel_spectrogram", "text", "cleaned_text", "wav_file"]

RANDOM_SEED = 42


class LJSpeech11Data(Dataset):
    def __init__(self, 
                 data_dir = "./", 
                 input_data_type = None, 
                 output_data_type = None,
                 compute_mel_spectrogram=False,
                 n_fft = 2048,
                 hop_length = 250,
                 win_length = None,
                 n_mels = 80,
                 fmin = 0.0,
                 fmax = None,):
        super().__init__()
        self.input_data_type = input_data_type
        self.output_data_type = output_data_type
        self.data_dir = data_dir
        self.wav_dir = data_dir + "wavs/"
        self.mel_dir = data_dir + "mels/"
        self.texts = pd.read_csv(data_dir + "metadata.csv", sep='|', header=None)
        self.valid_types = [input_data_type, output_data_type]

        self.num_utterances=13100
        self.sr = 22050 # in Hz, this is what we expect from LJSpeech

        if input_data_type == "mel_spectrogram" or output_data_type == "mel_spectrogram":
            ## I need to do some automated checks to ensure that the correct mel_spectrogram is created
            if compute_mel_spectrogram:
                os.makedirs(data_dir + "mels/", exist_ok=True)
                
                prepare_mels(wav_dir= data_dir + "wavs/", 
                             mel_dir= data_dir + "mels/",
                             utterances=self.texts.iloc[:, 0],
                             sr=self.sr,
                             n_fft=n_fft,
                             n_mels=n_mels,
                             fmin=fmin,
                             fmax=fmax,
                             hop_length=hop_length,
                             win_length = win_length if win_length is not None else n_fft,
                )
    
    def __len__(self):
        if self.input_data_type in VALID_DATA_TYPES_LJSPEECH and self.input_data_type in VALID_DATA_TYPES_LJSPEECH:
            return self.num_utterances
        return 0

    def __getitem__(self, index):
        in_item = None
        out_item = None

        data_type_check = "mel_spectrogram"
        if self.input_data_type == data_type_check or self.output_data_type == data_type_check:
            name = self.texts.iat[index, 0]
            data = np.load(self.mel_dir + name + ".mel.npy")
            if self.input_data_type is data_type_check:
                in_item = data
            if self.output_data_type is data_type_check:
                out_item = data
        
        
        data_type_check = "text"
        if self.input_data_type == data_type_check or self.output_data_type == data_type_check:
            data = self.texts.iat[index, 1]
            if self.input_data_type is data_type_check:
                in_item = data
            if self.output_data_type is data_type_check:
                out_item = data
        
        
        data_type_check = "cleaned_text"
        if self.input_data_type == data_type_check or self.output_data_type == data_type_check:
            data = self.texts.iat[index, 2]
            if self.input_data_type is data_type_check:
                in_item = data
            if self.output_data_type is data_type_check:
                out_item = data
        
        
        data_type_check = "wav_file"
        if self.input_data_type == data_type_check or self.output_data_type == data_type_check:
            name = self.texts.iat[index, 0]
            data, _ = librosa.load(self.wav_dir  + name + ".wav")
            if self.input_data_type is data_type_check:
                in_item = data
            if self.output_data_type is data_type_check:
                out_item = data
        
        return in_item, out_item


class LJSpeech11(L.LightningDataModule):
    def __init__(self, 
                 data_dir: str = "./",
                 n_fft = 2048,
                 hop_length = 250,
                 win_length = None,
                 n_mels = 80,
                 fmin = 0.0,
                 fmax = None,
                 train_split = 0.8,
                 train=False,
                 input_data="mel_spectrogram",
                 output_data="wav_file",
                 batch_size=32,
                 compute_mel_spectrogram=False,
                 njt=False,
                 n_train_workers=23,
                 n_val_workers=23,
                 n_test_workers=23,
                 n_predict_workers=23,
        ):
        super().__init__()
        self.data_dir = data_dir
    
        self.train_split_num = int(train_split * 13100)

        self.n_fft=n_fft
        self.hop_length=hop_length
        self.win_length=win_length
        self.n_mels=n_mels
        self.fmin=fmin
        self.fmax=fmax
        self.train=train
        self.input_data = input_data
        self.output_data = output_data
        self.compute_mel_spectrogram = compute_mel_spectrogram

        self.n_train_workers=n_train_workers
        self.n_val_workers=n_val_workers
        self.n_test_workers=n_test_workers
        self.n_predict_workers=n_predict_workers

        self.batch_size=batch_size
        self.njt=njt


    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            full_data = LJSpeech11Data(data_dir=self.data_dir,
                                       input_data_type=self.input_data,
                                       output_data_type=self.output_data,
                                       compute_mel_spectrogram=self.compute_mel_spectrogram,
                                       n_fft=self.n_fft,
                                       hop_length=self.hop_length,
                                       win_length=self.win_length,
                                       n_mels=self.n_mels,
                                       fmin=self.fmin,
                                       fmax=self.fmax,
            )
            self.LJ_train, self.LJ_val = random_split(
                full_data, [self.train_split_num, 13100 - self.train_split_num], generator=torch.Generator().manual_seed(RANDOM_SEED)
            )


        if stage == "test":
            self.LJ_test = LJSpeech11Data(data_dir=self.data_dir,
                                          input_data_type=self.input_data,
                                          output_data_type=self.output_data,
                                          compute_mel_spectrogram=self.compute_mel_spectrogram,
                                          n_fft=self.n_fft,
                                          hop_length=self.hop_length,
                                          win_length=self.win_length,
                                          n_mels=self.n_mels,
                                          fmin=self.fmin,
                                          fmax=self.fmax,
            )

        if stage == "test":
            self.LJ_predict = LJSpeech11Data(data_dir=self.data_dir,
                                          input_data_type=self.input_data,
                                          output_data_type=self.output_data,
                                          compute_mel_spectrogram=self.compute_mel_spectrogram,
                                          n_fft=self.n_fft,
                                          hop_length=self.hop_length,
                                          win_length=self.win_length,
                                          n_mels=self.n_mels,
                                          fmin=self.fmin,
                                          fmax=self.fmax,
            )

    def train_dataloader(self):
        return DataLoader(self.LJ_train, self.batch_size, collate_fn=partial(_collate_fn_for_NJT_Tensors, njt=self.njt),num_workers=self.n_train_workers)
    
    def val_dataloader(self):
        return DataLoader(self.LJ_val, self.batch_size, collate_fn=partial(_collate_fn_for_NJT_Tensors, njt=self.njt), num_workers=self.n_val_workers)
    
    def test_dataloader(self):
        return DataLoader(self.LJ_test, self.batch_size, collate_fn=partial(_collate_fn_for_NJT_Tensors, njt=self.njt), num_workers=self.n_test_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.LJ_predict, self.batch_size, collate_fn=partial(_collate_fn_for_NJT_Tensors, njt=self.njt), num_workers=self.n_predict_workers)
    
def _collate_fn_for_NJT_Tensors(batch, njt=False):
    sequences = [a for a,_ in batch]
    labels = [b for _,b in batch]

    sequences_NJT = torch.nested.nested_tensor(sequences, layout=torch.jagged)
    labels_NJT = torch.nested.nested_tensor(labels, layout=torch.jagged)

    if not njt:
        sequences_NJT = torch.nested.to_padded_tensor(sequences_NJT, 0)
        labels_NJT = torch.nested.to_padded_tensor(labels_NJT, 0)

    return sequences_NJT, labels_NJT