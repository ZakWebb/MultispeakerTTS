
import librosa
import numpy as np


def prepare_mels(wav_dir, mel_dir, utterances, sr, n_fft, n_mels, fmin, fmax, 
                 hop_length, win_length):
    # save mel metadata


    # Create Mel to use for inner product
    mel_filter = librosa.filters.mel(sr=sr, 
                                     n_fft=n_fft, 
                                     n_mels=n_mels, 
                                     fmin=fmin, 
                                     fmax=fmax)
    

    for utter_name in utterances:
        wav, sr = librosa.load(wav_dir  + utter_name + ".wav")
        mel = librosa.feature.melspectrogram(y=wav, 
                                       sr=sr, 
                                       S = mel_filter,
                                       n_fft=n_fft,
                                       hop_length=hop_length,
                                       win_length=win_length,
                                       )
        np.save(mel_dir + utter_name + ".mel",  np.array(mel))