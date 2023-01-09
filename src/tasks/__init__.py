from tasks.base_task import BaseTask
from tasks.preprocess import get_preprocessor
from tasks.text_cleaners import get_cleaner
from tasks.text_to_phoneme import get_t2p
from tasks.phoneme_to_mel import get_p2m
from tasks.vocoders import get_vocoder


_TASKS = {"preprocess": get_preprocessor,
          "cleaner": get_cleaner,
          "text_to_phoneme": get_t2p,
          "phoneme_to_mel": get_p2m,
          "vocoder": get_vocoder
          }

def get_task(config):
    if config.get('task') is None:
        raise ValueError("No task defined in config files")
    if _TASKS.get(config['task']) is not None:
        return _TASKS[config['task']]
    raise ValueError("{name} is not a defined task.")