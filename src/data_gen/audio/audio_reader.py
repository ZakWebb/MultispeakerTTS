from data_gen.audio.tools import get_mel_from_wav
from data_gen.audio.stft import TacotronSTFT
import librosa
import soundfile as sf
import numpy as np
import os
from scipy.io import wavfile
import torch

class AudioReader(object):
    def __init__(self, config):
        self.target_sr = config["sample_rate"]
        self.stft = TacotronSTFT(config["filter_length"], config["hop_length"], \
            config["window_length"], config["n_mel_channels"], self.target_sr)
        self.top_db = config.get("top_db")
        self.norm = config.get("norm")

    def load_wav(self, dir, name):
        self.audio, self.current_sr = librosa.load(os.path.join(dir, name + ".wav"))
        self.processed = False
        self.computed_mel = False

    
    def process_wav(self):
        if self.processed:
            return
        if self.current_sr != self.target_sr:
            self.audio = librosa.resample(self.audio, orig_sr=self.current_sr, target_sr=self.target_sr)
            self.current_sr = self.target_sr
        if self.top_db is not None:
            trimmed, _ = librosa.effects.trim(self.audio, top_db = self.top_db)

        self.processed = True
    
    def save_wav(self, dir, name):
        if self.norm:
            wav = self.audio / np.abs(self.audio).max()
        wav *= 32767
        # proposed by @dsmiller
        sf.write(os.path.join(dir, name + ".wav"), wav.astype(np.int16), self.current_sr)

    def save_mel(self, dir, name):
        if not self.computed_mel:
            audio = torch.from_numpy(self.audio)
            audio = audio[None, :]
            mel, _ = self.stft.mel_spectrogram(audio)
            self.mel = torch.squeeze(mel)
            self.computed_mel = True
        torch.save(self.mel, os.path.join(dir, name + ".mel"))
        