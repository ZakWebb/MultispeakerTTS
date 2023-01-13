import itertools
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer

from modules.HiFiGAN.HiFiGAN_model import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, discriminator_loss, feature_loss, generator_loss
from tasks.vocoders.base_vocoder import register_vocoder
from data_gen.audio.stft import TacotronSTFT

@register_vocoder
class HiFiGAN(LightningModule):
    def __init__(self, config):
        super(HiFiGAN, self).__init__()

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

        self.generator = Generator(config=config)
        self.period_discriminator = MultiPeriodDiscriminator()
        self.scale_disciminator = MultiScaleDiscriminator()

        self.stft = TacotronSTFT(self.filter_length, self.hop_length, self.win_length, self.n_mel_channels, self.sampling_rate, self.mel_fmin, self.mel_fmax)
    
    def forward(self, x):
        return self.generator(x)
    
    def training_step(self, batch, batch_idx, optimizer_idx) :
        true_mels, true_wav =  batch

        gen_wav = self.generator(true_mels)

        gen_mels = self.stft.mel_spectrogram(gen_wav)

        gen_df_r, gen_df_g, fmap_f_r, fmap_f_g = self.period_discriminator(true_wav, gen_wav)
        gen_ds_r, gen_ds_g, fmap_s_r, fmap_s_g = self.scale_disciminator(true_wav, gen_wav)
        
        # train generator
        if optimizer_idx == 0:
            loss_mel = F.l1_loss(true_mels, gen_mels) * 45 # This constant is annoying, but is included in the paper


            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            
            loss_gen_f, losses_gen_f = generator_loss(gen_df_g)
            loss_gen_s, losses_gen_s = generator_loss(gen_ds_g)

            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            return loss_gen_all

        # train discriminator
        if optimizer_idx == 1:
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(gen_df_r, gen_df_g)
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(gen_ds_r, gen_ds_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            return loss_disc_all
        
    
    def validation_step(self, batch, batch_idx):
        true_mels, _ =  batch

        gen_wavs = self.generator(true_mels)

        gen_mels = self.stft.mel_spectrogram(gen_wavs)

        val_err_tot = F.l1_loss(true_mels, gen_mels)
        self.log("val_loss", val_err_tot)
    
    def test_step(self, batch, batch_idx):
        true_mels, _ =  batch

        gen_wavs = self.generator(true_mels)

        gen_mels = self.stft.mel_spectrogram(gen_wavs)

        test_err_tot = F.l1_loss(true_mels, gen_mels)
        self.log("test_loss", test_err_tot)
    
    # def predict_step(self, batch, batch_idx):
    #     raise NotImplemented
    
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(itertools.chain(self.scale_disciminator.parameters(), self.period_discriminator.parameters()), \
            lr=self.lr, betas=(self.b1, self.b2))

        return [opt_g, opt_d], []