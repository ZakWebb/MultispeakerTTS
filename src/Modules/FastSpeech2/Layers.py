import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

    def forward(self, src_seq, mask):
        raise NotImplemented

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
    
    def forward():
        raise NotImplemented

class PreNet(nn.Module):
    def __init__(self, config):
        super(PreNet, self).__init__()

    def forward():
        raise NotImplemented

class VarianceAdaptor(nn.Module):
    def __init__(self, config):
        super(VarianceAdaptor, self).__init__()
