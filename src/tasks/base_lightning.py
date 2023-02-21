import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, TQDMProgressBar
from pytorch_lightning.profilers import AdvancedProfiler, PyTorchProfiler, SimpleProfiler
import torch
from torch.utils.data import DataLoader
import re
from glob import glob

from utils.lightning_callbacks import ProfCallback


from tasks.datasets import TTSDataset, get_TTSDataset_collater
from tasks.base_conversion import BaseConversion

class BaseLit(BaseConversion):
    def __init__(self, config):
        super(BaseLit, self).__init__(config)

        self.trainable = config["trainable"]
        self.training = config["train"]

        self.task = config["task"]

        self.ckpt_dir = os.path.join(config["work_dir"], config["task"], config[config["task"] + "_config"]["model_name"])
        self.ckpt = self.get_ckpt(config)
        self.load_ckpt = config.get("load_ckpt", False)

        self.num_workers = config["num_workers"]

        if config["use_gpu"]:
            self.accelerator = 'gpu'
            self.pin_memory = config.get("pin_memory", False)
        else:
            self.accelerator = 'cpu'
            self.pin_memory = False
        self.devices = 1

        self.accumulate_gradient = config.get("accumulate_gradient", 1)

        self.batch_size = config["batch_size"]
        self.shuffle = config["shuffle"]

        collate_fn = get_TTSDataset_collater(self.input, self.output)

        self.limit_predict_batches = config.get("limit_predict_batches", 1.0)
        self.max_epochs = config.get("max_epochs", -1)

        if self.trainable:
            self.train_loader = DataLoader(TTSDataset(config, self.input, self.output, "train"), self.batch_size, self.shuffle, num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=self.pin_memory, persistent_workers=True)
            self.valid_loader = DataLoader(TTSDataset(config, self.input, self.output, "valid"), self.batch_size, False, num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=self.pin_memory, persistent_workers=True)
            self.test_loader = DataLoader(TTSDataset(config, self.input, self.output, "test"), self.batch_size, False, num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=self.pin_memory, persistent_workers=True)
        else:
            self.train_loader = None
            self.valid_loader = None
            self.test_loader = None

        self.model = None

        self.callbacks = [#DeviceStatsMonitor(),
                                ProfCallback(),
                                TQDMProgressBar(refresh_rate=10)]

        if config.get("save_top_k") is not None:
            self.callbacks.append(ModelCheckpoint(save_top_k=config["save_top_k"],
                                        monitor="val_loss",
                                        mode="min",
                                        dirpath=self.ckpt_dir,
                                        filename=config[config["task"] + "_config"]["model_name"] + "-{epoch:02d}-{val_loss:.6f}"))
        
        self.callbacks.append(ModelCheckpoint(save_top_k=config.get("save_last_k",5),
                                    monitor="global_step",
                                    mode="max",
                                    dirpath=self.ckpt_dir,
                                    filename=config[config["task"] + "_config"]["model_name"] + "-{epoch:02d}-{val_loss:.6f}"))
                                        


        self.trainer = pl.Trainer(callbacks=self.callbacks, 
                                    accelerator=self.accelerator, 
                                    devices=self.devices,
                                    default_root_dir=self.ckpt_dir,
                                    accumulate_grad_batches=self.accumulate_gradient,
                                    log_every_n_steps=25,
                                    limit_predict_batches=self.limit_predict_batches,
                                    profiler=AdvancedProfiler(dirpath=self.ckpt_dir, filename="perf_logs"),
                                    max_epochs=self.max_epochs
                                )

    def train(self):
        if self.training:
            print("starting training")
            self.trainer.fit(self.model, self.train_loader, self.valid_loader, ckpt_path=self.ckpt)
#            
        else:
            print("starting testing")
            self.trainer.test(self.model, self.test_loader, ckpt_path=self.ckpt)
           
    
    def predict(self):
        return self.trainer.predict(self.model, self.test_loader, ckpt_path=self.ckpt)
    
    def get_ckpt(self, config):
        ckpt = config.get("ckpt")
        if ckpt is None:
            ckpts = glob(os.path.join(self.ckpt_dir, config[self.task + "_config"]["model_name"] + "*.ckpt"))
            
            model_name_ints = len(re.findall("\d+", config[self.task + "_config"]["model_name"]))

            def find_epoch(name):
                temp = os.path.basename(name)
                nums = re.findall("\d+", temp)
                if len(nums) > 0:
                    return int(nums[model_name_ints])
                else:
                    return -1

            epochs = list(map(find_epoch, ckpts))

            ckpts = sorted(zip(ckpts, epochs), key=lambda x: x[1], reverse=True)
            
            if len(ckpts) > 0:
                ckpt = ckpts[0][0]
        
        return ckpt

    def convert(self, input):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input)
        return output