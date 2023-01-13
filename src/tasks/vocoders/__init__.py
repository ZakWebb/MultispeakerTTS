#from tasks.vocoders import hifigan
#from tasks.vocoders import fastdiff
from tasks.vocoders.base_vocoder import VocoderTask, register_vocoder
from tasks.vocoders.hifigan import HiFiGAN

_VOCODERS = {HiFiGAN}
map(register_vocoder, _VOCODERS)


def get_vocoder(config):
    return VocoderTask