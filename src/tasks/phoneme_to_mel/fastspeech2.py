import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from tasks.phoneme_to_mel.base_phoneme_to_mel import register_p2m
from modules.FastSpeech2.FastSpeech2_model import FastSpeech2_model
from modules.FastSpeech2.Loss import StyleSpeechLoss


class FastSpeech2(LightningModule):
    def __init__(self, config):
        super(FastSpeech2, self).__init__()

        self.lr = config["learning_rate"]
        self.b1 = config["adam_b1"]
        self.b2 = config["adam_b2"]
        self.weight_decay = config.get("weight_decay", 0)

        self.filter_length = config["filter_length"]
        self.hop_length = config["hop_length"]
        self.win_length = config["window_length"]
        self.n_mel_channels = config["n_mel_channels"]
        self.sampling_rate = config["sample_rate"]
        self.mel_fmin = config["mel_fmin"]
        self.mel_fmax = config["mel_fmax"]

        self.p2m = FastSpeech2_model(config)

        self.loss = StyleSpeechLoss()

    def forward(self, batch):
        inputs = batch["inputs"]
        phonemes = inputs["data"]
        phoneme_masks = inputs["mask"]
        phoneme_lens = inputs["lens"]
        # phoneme_durs = inputs.get("durations", None)
        mel_prediction, src_embedded, d_prediction, mel_mask, mel_len = self.p2m(phonemes, phoneme_masks)

        return mel_prediction, mel_mask, mel_len

        
    
    def training_step(self, batch, batch_idx, optimizer_idx) :
        inputs = batch["inputs"]
        phonemes = inputs["data"]
        phoneme_lens = inputs["lens"]
        phoneme_masks = inputs["masks"]
        phoneme_durs = inputs["durations"]

        outputs = batch["outputs"]
        mel_true = outputs["data"]
        mel_masks_true = outputs["masks"]
        mel_lens_true = outputs["lens"]

        mel_predicted, src_output, log_duration_output, mel_mask, mel_len_predicted  = self.p2m(
                    phonemes, phoneme_masks)

        mel_loss, d_loss = self.loss(mel_predicted, mel_true, 
                    log_duration_output, phoneme_durs, phoneme_lens, mel_len_predicted) # zeroing logD, f0, and energy

        loss = mel_loss + d_loss 


        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs = batch["inputs"]
        phonemes = inputs["data"]
        phoneme_lens = inputs["lens"]
        phoneme_masks = inputs["masks"]
        phoneme_durs = inputs["durations"]

        outputs = batch["outputs"]
        mel_true = outputs["data"]
        mel_masks_true = outputs["masks"]
        mel_lens_true = outputs["lens"]

        mel_predicted, src_output, log_duration_output, mel_mask, mel_len_predicted  = self.p2m(
                    phonemes, phoneme_masks)

        mel_loss, d_loss = self.loss(mel_predicted, mel_true, 
                    log_duration_output, phoneme_durs, phoneme_lens, mel_len_predicted) # zeroing logD, f0, and energy

        loss = mel_loss + d_loss 


        self.log("val_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch["inputs"]
        phonemes = inputs["data"]
        phoneme_lens = inputs["lens"]
        phoneme_masks = inputs["masks"]
        phoneme_durs = inputs["durations"]

        outputs = batch["outputs"]
        mel_true = outputs["data"]
        mel_masks_true = outputs["masks"]
        mel_lens_true = outputs["lens"]

        mel_predicted, src_output, log_duration_output, mel_mask, mel_len_predicted  = self.p2m(
                    phonemes, phoneme_masks)

        mel_loss, d_loss = self.loss(mel_predicted, mel_true, 
                    log_duration_output, phoneme_durs, phoneme_lens, mel_len_predicted) # zeroing logD, f0, and energy

        loss = mel_loss + d_loss 


        self.log("test_loss", loss)

        return loss

    # def predict_step(self, batch, batch_idx):
    #     raise NotImplemented
    
    def configure_optimizers(self):
        self.opt = torch.optim.AdamW(
            self.p2m.parameters(),
            lr=self.lr, betas=(self.b1, self.b2), weight_decay=self.weight_decay)
        return [self.opt], []
    
    def training_epoch_end(self, training_step_outputs):
        self.log("global_step", self.global_step * 1.0)
        return super().training_epoch_end(training_step_outputs)