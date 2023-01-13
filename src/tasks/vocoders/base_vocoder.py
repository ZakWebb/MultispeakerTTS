import torch
from torch.utils.data import DataLoader
from glob import glob

from tasks.base_conversion import BaseConversion
from tasks.base_lightning import BaseLit
from tasks.base_task import BaseTask
VOCODERS = {}


def register_vocoder(cls):
    VOCODERS[cls.__name__.lower()] = cls
    VOCODERS[cls.__name__] = cls
    return cls


def get_vocoder_cls(config):
    if config['vocoder'] in VOCODERS:
        return VOCODERS[config['vocoder']]
    raise ValueError("Vocoder {name} is not a recognized vocoder.".format(name=config["vocoder"]))


class BaseVocoder(BaseLit):
    def __init__(self, config):
        self.input = "mels"
        self.output = "wavs"

        super(BaseVocoder, self).__init__(config)

        if not self.trainable and self.load_ckpt:
            self.model = get_vocoder_cls(config).load_from_checkpoint(self.ckpt)
        else:
            self.model = get_vocoder_cls(config)(config["vocoder_config"])

    
    
    def get_ckpt(self, config):
        ckpt = config.get("ckpt")
        if ckpt is None:
            model_name = get_vocoder_cls(config).__name__
            print(self.ckpt_dir + model_name)
            ckpts = sorted(glob(self.ckpt_dir + model_name + "*.pt"), reverse=True) # this might not actually yeild the latest checkpoint
            if len(ckpts) > 0:
                ckpt = ckpts[0]
        
        return ckpt

class VocoderTask(BaseTask):
    def __init__(self, config):
        super(VocoderTask, self).__init__()
        self.vocoder = BaseVocoder(config)
        self.train = config["train"]
    
    def start(self):
        if self.train:
            self.vocoder.train()
        