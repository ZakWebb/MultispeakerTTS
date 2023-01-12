import torch
from pytorch_lightning import LightningModule, Trainer

from modules.hifi-gan.hifi-gan_model import Generator, 

class HiFiGAN(LightningModule):
    def __init__(self, config):
        super(HiFiGAN).__init__()

        self.generator = G
    
    def forward(self, x):
        raise NotImplemented
    
    def training_step(self, batch, batch_idx, optimizer_idx) :
        raise NotImplemented
    
    def validation_step(self, batch, batch_idx):
        raise NotImplemented
    
    def test_step(self, batch, batch_idx):
        raise NotImplemented
    
    def predict_step(self, batch, batch_idx):
        raise NotImplemented
    
    def configure_optimizers(self):
        raise NotImplemented