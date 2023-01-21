import torch
from torch.utils.data import DataLoader
from glob import glob
import os

from tasks.base_conversion import BaseConversion
from tasks.base_lightning import BaseLit
from tasks.base_task import BaseTask
from data_gen.audio.audio_reader import AudioReader

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


        if self.trainable and self.load_ckpt and self.ckpt is not None and os.path.exists(self.ckpt):
            self.model = get_vocoder_cls(config).load_from_checkpoint(self.ckpt,config=config["vocoder_config"])
        else:
            self.model = get_vocoder_cls(config)(config["vocoder_config"])

    
    
    def get_ckpt(self, config):
        ckpt = config.get("ckpt")
        if ckpt is None:
            ckpts = sorted(glob(os.path.join(self.ckpt_dir, config["vocoder_config"]["model_name"] + "*.ckpt")), reverse=True) # this might not actually yeild the latest checkpoint
            if len(ckpts) > 0:
                ckpt = ckpts[0]
        
        return ckpt

class VocoderTask(BaseTask):
    def __init__(self, config):
        super(VocoderTask, self).__init__()
        self.vocoder = BaseVocoder(config)
        self.train = config["train"]
        
        self.sample_rate = config["sample_rate"]
        self.filter_length = config["filter_length"]
        self.hop_length = config["hop_length"]
        self.window_length = config["window_length"]
        self.n_mel_channels = config["n_mel_channels"]


    
    def start(self):
        if self.train:
            self.vocoder.train()
        else:
            #self.vocoder.train()
            #convert_torch_to_wavs(os.path.join())


            config = {"sample_rate": self.sample_rate,
                        "filter_length": self.filter_length,
                        "hop_length": self.hop_length,
                        "window_length": self.window_length,
                        "n_mel_channels": self.n_mel_channels
            }

            audio_reader = AudioReader(config)

            gen_wavs = self.vocoder.predict()

            wav_path = os.path.join(self.vocoder.ckpt_dir, "Step {} wavs".format(self.vocoder.model.global_step))
            os.makedirs(wav_path, exist_ok = True)

            i = 0
            for wavs in gen_wavs:
                for j in range(wavs.size(0)):
                    audio_reader.set_wav(torch.squeeze(wavs[i,:,:]).cpu().numpy(), self.sample_rate)
                    audio_reader.save_wav(wav_path, "{}".format(i))
                    i += 1
    
    
        