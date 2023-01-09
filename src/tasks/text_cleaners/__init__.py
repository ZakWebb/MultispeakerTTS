""" from https://github.com/keithito/tacotron """
import re
#from text_cleaners import cleaners
#from text_cleaners.symbols import symbols
from tasks.text_cleaners.base_text_cleaner import BaseTextCleaner, get_cleaner_cls
from tasks.text_cleaners.simple_english_cleaner import SimpleEnglishCleaner

_CLEANERS = {"Simple English": SimpleEnglishCleaner}

def get_cleaner(config):
    if config.get("cleaner") is None:
        raise ValueError("No defined cleaner in config file.")
    name = config["cleaner"]
    if _CLEANERS.get(name) is not None:
        return _CLEANERS[name]
    raise ValueError("{name} is not a defined cleaner.")