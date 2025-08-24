import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal

import lightning as pl

class SimpleVocoder(pl.LightningModule):
    def __init__(self,
                 n_mels=80,
                 d_model=512,
                 lr=1e-3):
        super().__init__()

        self.n_mels=n_mels
        self.d_model = d_model,
        self.lr = lr

        loc = (torch.tensor(0.0)).to(self.device)
        scale = (torch.tensor(np.sqrt(0.5))).to(self.device)

        self.normal = Normal(loc,scale)

        self.in_layer = nn.Linear(in_features=n_mels, out_features=d_model)
        self.out_layer = nn.Linear(in_features=d_model, out_features=1)

    def forward(self, x, wav_file=None):
        x = torch.transpose( x, -2, -1)
        x = self.in_layer(x)
        x = torch.transpose( x, -2, -1)

        x = torch.transpose(x, -2, -1)
        x = self.out_layer(x)
        x = torch.flatten(x, -2, -1)

        return x


    # Figure out a better optimizer
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr) # pyright: ignore[reportAttributeAccessIssue]
        return optimizer
    


    def training_step(self, batch, batch_idx):
        mel_input, wav_real = batch


        output_wav = self.forward(mel_input)
        likelihood = torch.sum(self.normal.log_prob(output_wav), (-2, -1))

        temp1 = torch.nn.functional.pad(output_wav, (0, wav_real.shape[-1] - output_wav.shape[-1]), "constant", 0.0)


        temp = torch.mean((temp1 * wav_real), dim=-1)

        return -(likelihood +  temp).mean()


    def validation_step(self, batch, **kwargs):
        mel_input, wav_real = batch


        output_wav = self.forward(mel_input)
        likelihood = torch.sum(self.normal.log_prob(output_wav), (-2, -1))

        temp1 = torch.nn.functional.pad(output_wav, (0, wav_real.shape[-1] - output_wav.shape[-1]), "constant", 0.0)


        temp = torch.mean((temp1 * wav_real), dim=-1)

        return -(likelihood +  temp).mean()
    
    def test_step(self, batch, batch_idx):
        raise NotImplementedError