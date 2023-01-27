from tasks.phoneme_to_mel.base_phoneme_to_mel import Phoneme2MelTask, register_p2m
_P2M = {}

map(register_p2m, _P2M)

def get_p2m(config):
    return Phoneme2MelTask