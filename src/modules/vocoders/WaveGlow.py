import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from ..components import FlowStep, SqueezeLayer


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