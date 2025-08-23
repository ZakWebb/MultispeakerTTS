import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

import lightning as pl



from ..components import FlowStep, SqueezeLayer, UpsampleNet


class WaveGlow(nn.Module):
    """Implements the WaveGlow model."""

    def __init__(self,
                 squeeze_factor=8,
                 num_layers=12,
                 wn_filter_width=3,
                 wn_dilation_layers=8,
                 wn_residual_channels=512,
                 wn_dilation_channels=256,
                 wn_skip_channels=256,
                 local_condition_channels=None):
        """Initializes the WaveGlow model.

        Args:
            local_condition_channels: Number of channels in local conditioning
                vector. None indicates there is no local conditioning.
        """
        super(WaveGlow, self).__init__()

        self.squeeze_factor = squeeze_factor
        self.num_layers = num_layers
        self.num_scales = squeeze_factor // 2

        self.squeeze_layer = SqueezeLayer(squeeze_factor)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                FlowStep(squeeze_factor,
                         wn_filter_width=wn_filter_width,
                         wn_dilation_layers=wn_dilation_layers,
                         wn_residual_channels=wn_residual_channels,
                         wn_dilation_channels=wn_dilation_channels,
                         wn_skip_channels=wn_skip_channels,
                         local_condition_channels=local_condition_channels))
            # Use multi-scale architecture to output 2 of the channels
            # after every 4 coupling layers.
            if (i + 1) % self.num_scales == 0:
                squeeze_factor -= 2

    def forward(self, input, logdet, reverse, local_condition):
        if not reverse:
            output, logdet = self.squeeze_layer(input, logdet=logdet, rerverse=False)

            early_outputs = []
            for i, layer in enumerate(self.layers):
                output, logdet = layer(output, logdet=logdet, reverse=False,
                                       local_condition=local_condition)

                if (i + 1) % self.num_scales == 0:
                    early_output, output = output.split([2, output.size(1) - 2], 1)
                    early_outputs.append(early_output)
            early_outputs.append(output)

            return torch.cat(early_outputs, 1), logdet
        else:
            output = input
            for i, layer in enumerate(reversed(self.layers)):
                curr_input = output[:, -2 * (i // self.num_scales + 2):, :]
                curr_output, logdet = layer(curr_input, logdet=logdet, reverse=True,
                                            local_condition=local_condition)
                output[:, -2 * (i // self.num_scales + 2):, :] = curr_output

            output, logdet = self.squeeze_layer(output, logdet=logdet, reverse=True)

            return output, logdet



class WaveGlowLightning(pl.LightningModule):
    def __init__(self,
                 squeeze_factor=8,
                 num_layers=12,
                 wn_filter_width=3,
                 wn_dilation_layers=8,
                 wn_residual_channels=512,
                 wn_dilation_channels=256,
                 wn_skip_channels=256,
                 local_condition_channels=None):
        super().__init__()

        self.squeeze_factor = squeeze_factor
        self.num_layers = num_layers
        self.num_scales = squeeze_factor // 2
        self.num_layers=num_layers,
        self.wn_filter_width=wn_filter_width,
        self.wn_dilation_layers=wn_dilation_layers,
        self.wn_residual_channels=wn_residual_channels,
        self.wn_dilation_channels=wn_dilation_channels,
        self.wn_skip_channels=wn_skip_channels,
        self.local_condition_channels=local_condition_channels


        self.model = WaveGlow(squeeze_factor=squeeze_factor,
                 num_layers=num_layers,
                 wn_filter_width=wn_filter_width,
                 wn_dilation_layers=wn_dilation_layers,
                 wn_residual_channels=wn_residual_channels,
                 wn_dilation_channels=wn_dilation_channels,
                 wn_skip_channels=wn_skip_channels,
                 local_condition_channels=local_condition_channels)

        self.upsampler = UpsampleNet(upsample_factor=200,
                                     upsample_method="duplicate",
                                     squeeze_factor=squeeze_factor)

    def forward(self, x, logdet, reverse, local_condition):
        return self.model(x, logdet, reverse, local_condition)


    # Figure out a better optimizer
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr) # pyright: ignore[reportAttributeAccessIssue]

        # We don't return the lr scheduler becasue we need to apply it per iteration, not per epoch
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_iters) # pyright: ignore[reportAttributeAccessIssue]
        
        return optimizer
    

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_closure= None):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.lr_scheduler.step()

    def training_step(self, batch, batch_idx):
        mel_input, wav_real = batch

        wav_real = self.upsampler(wav_real)
        logdet = torch.zeros_like(mel_input[:,0,0])
        output_wav, logdet = self.model(mel_input, logdet=logdet,reverse=False,local_condition=wav_real)

        likelihood = torch.sum(normal.log_prob(output_wav), (1,2))

        return -(likelihood + logdet).mean()
    
    def validation_step(self, *args, **kwargs):
        raise NotImplementedError
    
    def test_step(self, batch, batch_idx):
        raise NotImplementedError