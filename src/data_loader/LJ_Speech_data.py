import os
import numpy as np
import pandas as pd
import librosa


import lightning as L
from torch.utils.data import random_split, DataLoader

class LJSpeech11(L.LightningDataModule):
    def __init__(self, 
                 data_dir: str = "./",
                 n_fft = 2048,
                 hop_length = 250,
                 win_length = None,
                 mel_center = True,
                 n_mels = 80,
                 fmin = 0.0,
                 fmax = None
        ):
        super().__init__()
        self.data_dir = data_dir
        self.audio_dir = data_dir + "wavs/"
        self.mel_dir = data_dir + "mels/"
        self.num_utterances = 13100 # Something that should be true for data
    
        self.texts = pd.read_csv(data_dir + "metadata.csv", sep='|')

        self.sr = 22050 # in Hz, this is what we expect from LJSpeech

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.mel_center = mel_center
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        
        self._prepare_mels()
    
    def _prepare_mels(self):
        # save mel metadata
        mel_filter = librosa.filters.mel(sr=self.sr, 
                                         n_fft=self.n_fft, 
                                         n_mels=self.n_mels, 
                                         fmin=self.fmin, 
                                         fmax=self.fmax)
        for i in range(self.num_utterances):
            wav, sr = librosa.load(self.audio_dir  + self.texts.iat[[i,0]] + ".wav")
            mel = librosa.feature.melspectrogram(y=wav, 
                                           sr=self.sr, 
                                           S = mel_filter,
                                           n_fft=self.n_fft,
                                           hop_length=self.hop_length,
                                           win_length=self.win_length,
                                           center=self.mel_center,
                                           )
            np.save(self.mel_dir + self.texts.iat[[i,0]] + ".mel",  np.array(mel))
                



#   def prepare_data(self):

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
