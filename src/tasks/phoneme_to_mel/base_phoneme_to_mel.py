import os

from tasks.base_task import BaseTask


_P2M = {}


def register_p2m(cls):
    _P2M[cls.__name__.lower()] = cls
    _P2M[cls.__name__] = cls
    return cls


def get_p2m_cls(config):
    if config['p2m'] in _P2M:
        return _P2M[config['p2m']]
    raise ValueError("Phoneme-to-mel {name} is not a recognized converter.".format(name=config["p2m"]))


class BasePhoneme2Mel(BaseTask):
    def __init__(self, config):
        self.input = "phonemes"
        self.output = "mels"

        super(BasePhoneme2Mel, self).__init__(config)


        if self.trainable and self.load_ckpt and self.ckpt is not None and os.path.exists(self.ckpt):
            self.model = get_p2m_cls(config).load_from_checkpoint(self.ckpt,config=config["p2m_config"])
        else:
            self.model = get_p2m_cls(config)(config["p2m_config"])


class Phoneme2MelTask(BaseTask):
    def __init__(self, config):
        super(Phoneme2MelTask, self).__init__()
        self.p2m = BasePhoneme2Mel(config)
        self.train = config["train"]
        
    
    def start(self):
        if self.train:
            self.p2m.train()
        else:
            self.p2m.test()
        