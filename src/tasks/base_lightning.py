import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from tasks.datasets import TTSDataset
from tasks.base_conversion import BaseConversion

class BaseLit(BaseConversion):
    def __init__(self, config):
        super(BaseLit, self).__init__(config)

        self.trainable = config["trainable"]

        self.ckpt_dir = config["work_dir"]
        self.ckpt = self.get_ckpt(config)

        
        self.batch_size = config["batch_size"]
        self.shuffle = config["shuffle"]

        if self.trainable:
            self.train_loader = DataLoader(TTSDataset(config, "mels", "wavs", "train"), self.batch_size, self.shuffle)
            self.valid_loader = DataLoader(TTSDataset(config, "mels", "wavs", "valid"), self.batch_size, False)
            self.test_loader = DataLoader(TTSDataset(config, "mels", "wavs", "test"), self.batch_size, False)
        else:
            self.train_loader = None
            self.valid_loader = None
            self.test_loader = None

        self.model = None

        self.trainer = pl.Trainer(default_root_dir= self.ckpt_dir)

    def train(self):
        if self.trainable:
            self.trainer.fit(self.model, self.train_loader, self.valid_loader, ckpt_path=self.ckpt)
            self.trainer.test(self.model, self.test_loader)
        else:
            ValueError("Trying to train model {self.model.name} when not trainable.")
    
    def get_ckpt(self, config):
        return config.get("ckpt")

    def convert(self, input):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input)
        return output