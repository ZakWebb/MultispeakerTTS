import torch
from torch import nn


# Copied from the github npuichigo/waveglow repo
# I haven't checked it, because I don't really understand the math behind
# WaveGlow yet
class GatedDilatedConv1d(nn.Module):
    """Creates a single causal dilated convolution layer.

    The layer contains a gated filter that connects to dense output
    and to a skip connection:
           |-> [gate]   -|        |-> 1x1 conv -> skip output
           |             |-> (*) -|
    input -|-> [filter] -|        |-> 1x1 conv -|
           |                                    |-> (+) -> dense output
           |------------------------------------|
    Where `[gate]` and `[filter]` are causal convolutions with a
    non-linear activation at the output. Biases and global conditioning
    are omitted due to the limits of ASCII art.
    """

    def __init__(self,
                 filter_width,
                 dilation,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 local_condition_channels=None):
        """Initializes the GatedDilatedConv1d.

        Args:
            filter_width:
            dilation:
            residual_channels:
            dilation_channels:
            skip_channels:
            local_condition_channels:
        """
        super(GatedDilatedConv1d, self).__init__()
        self.filter_width = filter_width
        self.dilation = dilation
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.local_condition_channels = local_condition_channels

        if filter_width % 2 == 0:
            raise ValueError("You must specify an odd number to filter_width "
                             "to make sure the shape is invariant after conv.")
        padding = (filter_width - 1) // 2 * dilation
        self.conv_filter_gate = nn.Conv1d(residual_channels,
                                          dilation_channels * 2,
                                          filter_width, padding=padding,
                                          dilation=dilation)

        if local_condition_channels is not None:
            self.conv_lc_filter_gate = nn.Conv1d(local_condition_channels,
                                                 dilation_channels * 2, 1,
                                                 bias=False)

        # The 1x1 conv to produce the residual output
        self.conv_dense = nn.Conv1d(dilation_channels, residual_channels, 1)

        # The 1x1 conv to produce the skip output
        self.conv_skip = nn.Conv1d(dilation_channels, skip_channels, 1)

    def forward(self, sample, local_condition):
        """
        Args:
            sample: Shape: [batch_size, channels, time].
            local_condition: Shape: [batch_size, channels, time].
        """
        sample_filter_gate = self.conv_filter_gate(sample)

        if self.local_condition_channels is not None:
            lc_filter_gate = self.conv_lc_filter_gate(local_condition)
            sample_filter_gate += lc_filter_gate

        sample_filter, sample_gate = torch.split(
            sample_filter_gate, self.dilation_channels, 1)
        gated_sample_batch = torch.tanh(sample_filter) * torch.sigmoid(sample_gate)

        # The 1x1 conv to produce the residual output
        transformed = self.conv_dense(gated_sample_batch)
        residual_output = transformed + sample

        # The 1x1 conv to produce the skip output
        skip_output = self.conv_skip(gated_sample_batch)
        return residual_output, skip_output



class WaveNet(nn.Module):
    """Implements the WaveNet block for WaveGlow.
    
    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2 # Convolutions just use 2 samples
        residual_channels = 16 # Not specified in the paper
        dilation_channels = 32 # Not specified in the paper
        skip_channels = 16 # Not specified in the paper
        net = WaveNet(dilations, 
                      filter_width,
                      residual_channels,
                      dilation_channels,
                      skip_channels)
        output_batch = net(input_batch)
    
    """

    def __init__(self,
                 filter_width,
                 dilations,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 input_channels,
                 output_channels,
                 local_condition_channels=None,
    ):
        """Initializes the WaveNet model.
        
        Args:
            filter_width: The samples that are included in each convolution,
                after dilating.
            dilations: A list with the dilation factor for each layer.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn fr the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the 
                quantized softmax output.
            input_channels:
            outpu_channels:
            local_condition_channels: Number of channels in local conditioning
                vector.  None indicates that the is no local conditioning.
        """
        super(WaveNet, self).__init__()
        self.filter_width = filter_width
        self.dilations = dilations
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.local_condition_channels = local_condition_channels

        # Create the conv layer for channel transformation
        self.preprocessing_layer = nn.Conv1d(input_channels, residual_channels, 1)
        
        # Creates the dilated conv layers
        self.dilated_conv_layers = nn.ModuleList([
            GatedDilatedConv1d(
                filter_width=filter_width,
                dilation=dilation,
                residual_channels=residual_channels,
                dilation_channels=dilation_channels,
                skip_channels=skip_channels,
                local_condition_channels=local_condition_channels
            )
            for dilation in dilations]
        )

        # Performs (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
        # postprocess the output.
        self.postprocessing_layers = nn.ModuleList([
            nn.ReLU(inplace=True),
            nn.Conv1d(skip_channels, skip_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(skip_channels, self.output_channels, 1)
        ])

    def forward(self, sample, local_condition):
        current_layer = self.preprocessing_layer(sample)

        skip_outputs = []
        for dilated_conv_layer in self.dilated_conv_layers:
            current_layer, skip_output = dilated_conv_layer(current_layer,
                                                            local_condition)
            skip_outputs.append(skip_output)

        # Adds up skip connections from the outputs of each layer.
        current_layer = sum(skip_outputs)

        for postprocessing_layer in self.postprocessing_layers:
            current_layer = postprocessing_layer(current_layer)

        return current_layer
    


class WaveNetVocoder(nn.Module):
    """Implements the WaveNet block for WaveGlow.
    
    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2 # Convolutions just use 2 samples
            Something weird is going on here, the code throws
            an error if filter_width is even.
        residual_channels = 16 # Not specified in the paper
        dilation_channels = 32 # Not specified in the paper
        skip_channels = 16 # Not specified in the paper
        net = WaveNet(dilations, 
                      filter_width,
                      residual_channels,
                      dilation_channels,
                      skip_channels)
        output_batch = net(input_batch)
    
    """

    def __init__(self,
                 filter_width=2,
                 dilation_power=2,
                 dilation_max_power=8,
                 dilation_depth=12,
                 residual_channels=16,
                 dilation_channels=32,
                 skip_channels=16,
                 n_mels=80,
                 local_condition_channels=None,
    ):
        """Initializes the WaveNet model.
        
        Args:
            filter_width: The samples that are included in each convolution,
                after dilating.
            dilations: A list with the dilation factor for each layer.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn fr the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the 
                quantized softmax output.
            input_channels:
            outpu_channels:
            local_condition_channels: Number of channels in local conditioning
                vector.  None indicates that the is no local conditioning.
        """
        super(WaveNetVocoder, self).__init__()

        dilations = [(dilation_power**i) for i in range(dilation_max_power)] * dilation_depth

        self.net = WaveNet(
                 filter_width=2 * filter_width - 1,
                 dilations=dilations,
                 residual_channels=residual_channels,
                 dilation_channels=dilation_channels,
                 skip_channels=skip_channels,
                 input_channels=n_mels,
                 output_channels=1,
                 local_condition_channels=local_condition_channels
        )

    def forward(self, sample, local_condition=None):
        output = self.net(sample, local_condition)

        return torch.flatten(output, -2, -1)