from tasks.vocoders.base_vocoder import VocoderTask, register_vocoder
from tasks.vocoders.hifigan import HiFiGAN
from tasks.vocoders.fastdiff import FastDiff

_VOCODERS = {HiFiGAN, FastDiff}
map(register_vocoder, _VOCODERS)


def get_vocoder(config):
    return VocoderTask