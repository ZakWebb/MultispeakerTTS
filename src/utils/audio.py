
import librosa
import numpy as np


def prepare_mels(wav_dir, mel_dir, utterances, sr, n_fft, n_mels, fmin, fmax, 
                 hop_length, win_length):
    # save mel metadata
    

    for utter_name in utterances:
        wav, sr = librosa.load(wav_dir  + utter_name + ".wav")
        mel = librosa.feature.melspectrogram(y=wav, 
                                       sr=sr, 
                                       n_mels=n_mels,
                                       n_fft=n_fft,
                                       hop_length=hop_length,
                                       win_length=win_length,
                                       )
        np.save(mel_dir + utter_name + ".mel",  np.array(mel))