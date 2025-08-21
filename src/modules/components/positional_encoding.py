# Standard Libraries
import math

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MAX_LENGTH_DEFAULT_VALUE = 5000

class PostionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LENGTH_DEFAULT_VALUE):
        """Positional Encoding.
        
        Args:
            d_model: Hidden dimensionality of the input.
            max_len: Maximum lenght of a sequence to expect.
        """
        super().__init__()

        # Create a matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(MAX_LENGTH_DEFAULT_VALUE * 2.0) / d_model))

        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # register_buffer => Tensor which is not a paramerter, but should be a part of the module's state.
        # Used for tensors that need to be on the same device as the module
        # persistent=False tells Pytorch to not add the buffer to the state dict (e.g., when we save the model)

        self.register_buffer("pos_enc", pos_enc, persistent=False)

    def forward(self, x):
        assert isinstance(self.pos_enc, torch.Tensor)
        x = x + self.pos_enc[:, : x.size(1)]
        return x