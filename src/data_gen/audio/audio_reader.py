from data_gen.audio.tools import get_mel_from_wav
from data_gen.audio.stft import TacotronSTFT
import librosa
import soundfile as sf
import numpy as np
import pyworld as pw
import os
from scipy.io import wavfile
from scipy.interpolate import interp1d
import torch

class AudioReader(object):
    def __init__(self, config):
        self.target_sr = config["sample_rate"]
        self.stft = TacotronSTFT(config["filter_length"], config["hop_length"], \
            config["window_length"], config["n_mel_channels"], self.target_sr)
        self.top_db = config.get("top_db")
        self.norm = config.get("norm")
        self.hop_length = config["hop_length"]
        self.audio = np.empty(0)
        self.processed = True
        self.computed_mel = False
        self.computed_f0 = False

    def load_wav(self, dir, name):
        self.audio, self.current_sr = librosa.load(os.path.join(dir, name + ".wav"), sr=self.target_sr)
        self.processed = False
        self.computed_mel = False
        self.computed_f0 = False

    def get_wav(self):
        return self.audio

    def set_wav(self, audio, sr):
        self.audio = audio
        self.current_sr = sr
    
    def process_wav(self):
        if self.processed:
            return
        if self.top_db is not None:
            trimmed, _ = librosa.effects.trim(self.audio, top_db = self.top_db)
        self.audio /= 32767

        self.processed = True
    
    def save_wav(self, dir, name):
        wav = self.audio
        if self.norm:
            wav = wav / np.abs(self.audio).max()
        wav *= 32767
        # proposed by @dsmiller
        sf.write(os.path.join(dir, name + ".wav"), wav.astype(np.int16), self.current_sr)

    def compute_mel(self):
        if not self.processed:
            self.process_wav()
        if self.computed_mel:
            return
        audio = torch.from_numpy(self.audio)
        audio = audio[None, None, :]
        mel, energy = self.stft.mel_spectrogram(audio)
        self.mel = torch.squeeze(mel)
        self.energy = torch.squeeze(energy)
        self.computed_mel = True
    
    # this is mostly a copy from StyleSpeech,  We probably want to go through this and actually understand it
    def compute_f0(self):
        if not self.processed:
            self.process_wav()
        audio = self.audio
        if self.computed_f0:
            return

        _f0, t = pw.dio(audio.astype(np.float64), self.current_sr, frame_period=self.hop_length/self.current_sr*1000)
        f0 = pw.stonemask(audio.astype(np.float64), _f0, t, self.current_sr)

        nonzero_ids = np.where(f0 != 0)[0]
        if len(nonzero_ids)>=2:
            interp_fn = interp1d(
                nonzero_ids,
                f0[nonzero_ids],
                fill_value=(f0[nonzero_ids[0]], f0[nonzero_ids[-1]]),
                bounds_error=False,
            )
            f0 = interp_fn(np.arange(0, len(f0)))
        
        self.f0 = torch.tensor(f0)
        self.computed_f0 = True

    def save_energy(self, dir, name):
        if not self.computed_mel:
            self.compute_mel()
        torch.save(self.energy, os.path.join(dir, name + ".energy"))

    def save_mel(self, dir, name):
        if not self.computed_mel:
            self.compute_mel()
        torch.save(self.mel, os.path.join(dir, name + ".mel"))
    
    def save_f0(self, dir, name):
        if not self.computed_f0:
            self.compute_f0()
        torch.save(self.f0, os.path.join(dir, name + ".f0"))
        
