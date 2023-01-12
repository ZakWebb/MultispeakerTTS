import torch
from glob import glob

from tasks.base_conversion import BaseConversion
from tasks.base_lightning import BaseLit
VOCODERS = {}


def register_vocoder(cls):
    VOCODERS[cls.__name__.lower()] = cls
    VOCODERS[cls.__name__] = cls
    return cls


def get_vocoder_cls(config):
    if config['vocoder'] in VOCODERS:
        return VOCODERS[config['vocoder']]
    name = config['vocoder']
    raise ValueError("Vocoder {name} is not a recognized vocoder.")


class BaseVocoder(BaseConversion, BaseLit):
    def __init__(self, config):
        super(self).__init__(config)

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        if not self.trainable and self.load_ckpt:
            self.model = get_vocoder_cls(config).load_from_checkpoint(self.ckpt)
        else:
            self.model = get_vocoder_cls(config)(config)

    def convert(self, phonemes):
        self.model.eval()
        with torch.no_grad():
            output = self.model(phonemes)
        return output
    
    def get_ckpt(self, config):
        ckpt = config.get("ckpt")
        if ckpt is None:
            model_name = get_vocoder_cls(config).__name__
            print(self.ckpt_dir + model_name)
            ckpts = sorted(glob(self.ckpt_dir + model_name + "*.pt"), reverse=True) # this might not actually yeild the latest checkpoint
            if len(ckpts) > 0:
                ckpt = ckpts[0]
        
        return ckpt