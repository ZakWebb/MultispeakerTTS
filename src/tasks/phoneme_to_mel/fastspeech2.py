import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from tasks.phoneme_to_mel.base_phoneme_to_mel import register_p2m
from modules.FastSpeech2.FastSpeech2_model import FastSpeech2_model


class FastSpeech2(LightningModule):
    def __init__(self, config):
        super(FastSpeech2, self).__init__()

        self.lr = config["learning_rate"]
        self.b1 = config["adam_b1"]
        self.b2 = config["adam_b2"]

        self.filter_length = config["filter_length"]
        self.hop_length = config["hop_length"]
        self.win_length = config["window_length"]
        self.n_mel_channels = config["n_mel_channels"]
        self.sampling_rate = config["sample_rate"]
        self.mel_fmin = config["mel_fmin"]
        self.mel_fmax = config["mel_fmax"]

        self.p2m = FastSpeech2_model(config)

    def forward(self, batch):
        phonemes, _ = batch
        return self.p2m(phonemes)
    
    def training_step(self, batch, batch_idx, optimizer_idx) :
    
    def validation_step(self, batch, batch_idx):

    def test_step(self, batch, batch_idx):

    # def predict_step(self, batch, batch_idx):
    #     raise NotImplemented
    
    def configure_optimizers(self):
    
    def training_epoch_end(self, training_step_outputs):
        self.log("global_step", self.global_step * 1.0)
        return super().training_epoch_end(training_step_outputs)