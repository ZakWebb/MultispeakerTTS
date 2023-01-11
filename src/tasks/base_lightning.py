import pytorch_lightning as pl
import torch

class BaseLit(object):
    def __init__(self, config):
        super(BaseLit, self).__init__()

        self.trainable = config["trainable"]

        self.ckpt_dir = config["work_dir"]
        self.ckpt = self.get_ckpt(config)
        
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