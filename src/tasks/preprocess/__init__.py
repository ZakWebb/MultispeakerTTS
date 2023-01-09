from tasks.preprocess.base_preprocess_task import BasePreprocessTask
from tasks.preprocess.LJSpeech_preprocess import LJSpeechPreprocess

_PREPROCESSORS = {"LJSpeech": LJSpeechPreprocess}

def get_preprocessor(config):
    if config.get("preprocessor") is None:
        raise ValueError("No defined preprocessor in config file.")
    name = config["preprocessor"]
    if _PREPROCESSORS.get(name) is not None:
        return _PREPROCESSORS[name]
    raise ValueError("{name} is not a defined preprocessor.")