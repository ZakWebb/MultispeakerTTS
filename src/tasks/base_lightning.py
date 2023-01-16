import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader


from tasks.datasets import TTSDataset, get_TTSDataset_collater
from tasks.base_conversion import BaseConversion

class BaseLit(BaseConversion):
    def __init__(self, config):
        super(BaseLit, self).__init__(config)

        self.trainable = config["trainable"]

        self.ckpt_dir = os.path.join(config["work_dir"], config["task"], config[config["task"] + "_config"]["model_name"])
        self.ckpt = self.get_ckpt(config)

        self.num_workers = config["num_workers"]

        if config["use_gpu"]:
            self.accelerator = 'gpu'
            self.pin_memory = config.get("pin_memory", False)
        else:
            self.accelerator = 'cpu'
            self.pin_memory = False
        self.devices = 1


        
        self.batch_size = config["batch_size"]
        self.shuffle = config["shuffle"]

        collate_fn = get_TTSDataset_collater(self.input, self.output)

        if self.trainable:
            self.train_loader = DataLoader(TTSDataset(config, self.input, self.output, "train"), self.batch_size, self.shuffle, num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=self.pin_memory)
            self.valid_loader = DataLoader(TTSDataset(config, self.input, self.output, "valid"), self.batch_size, False, num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=self.pin_memory)
            self.test_loader = DataLoader(TTSDataset(config, self.input, self.output, "test"), self.batch_size, False, num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=self.pin_memory)
        else:
            self.train_loader = None
            self.valid_loader = None
            self.test_loader = None

        self.model = None

        self.ckpt_callbacks = []

        if config.get("save_top_k") is not None:
            self.ckpt_callbacks.append(ModelCheckpoint(save_top_k=config["save_top_k"],
                                        monitor="val_loss",
                                        mode="min",
                                        dirpath=self.ckpt_dir,
                                        filename=config[config["task"] + "_config"]["model_name"] + "-{epoch:02d}-{val_loss:.2f}"))
        
        self.ckpt_callbacks.append(ModelCheckpoint(save_top_k=config.get("save_last_k",5),
                                    monitor="global_step",
                                    mode="max",
                                    dirpath=self.ckpt_dir,
                                    filename=config[config["task"] + "_config"]["model_name"] + "-{epoch:02d}-{val_loss:.2f}"))
                                        


        self.trainer = pl.Trainer(callbacks=self.ckpt_callbacks, accelerator=self.accelerator, devices=self.devices,default_root_dir=self.ckpt_dir,profiler="Simple")

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