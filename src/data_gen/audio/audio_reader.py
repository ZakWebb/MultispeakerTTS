from .tools import get_mel_from_wav
from .stft import TacotronSTFT
import librosa
import numpy as np
import os
from scipy.io import wavfile

class AudioReader(object):
    def __init__(self, config):
        self.target_sr = config["sample rate"]
        self.stft = TacotronSTFT(config["filter length"], config["hop length"], \
            config["win_length"], config["n_mel_channels"], self.target_sr)
        self.top_db = config.get("top db")

    def load_wav(self, dir, name):
        self.audio, self.current_sr = librosa.load(os.join(dir, name + ".wav"))
        self.processed = False
        self.computed_mel = False

    
    def process_wav(self):
        if self.processed:
            return
        if self.current_sr != self.target_sr:
            self.audio = librosa.resample(self.audio, self.current_sr, self.target_sr)
            self.current_sr = self.target_sr
        if self.top_db is not None:
            trimmed, _ = librosa.effects.trim(self.audio, top_db = self.top_db)

        self.processed = True
    
    def save_wav(self, dir, name):
        if self.norm:
            wav = self.audio / np.abs(self.audio).max()
        wav *= 32767
        # proposed by @dsmiller
        wavfile.write(os.join(dir, name + ".wav"), self.current_sr, wav.astype(np.int16))

    def save_mel(self, dir, name):
        if not self.computed_mel:
            self.mel = self.stft.mel_spectrogram(self.audio)
            self.computed_mel = True
        np.save(os.join(dir, name), self.mel)
        
