import csv
import librosa
import numpy as np
import os
import random
import subprocess

# Ideally, this should work for any language from common voice
# We're currently using English in order to test things out
data_dir = "/mnt/d/Speech Data/CV"
current_language = "en"  # English
window_length = 25  # length of spectrogram window in ms
hop_length = 10  # length of spectrogram window hop in ms
mfcc_elems = 40  # number of mfcc coefficents to use
encoder_utterance_length = 800  # length of utterance used for encoder


class Speaker():
    def __init__(self, data, create_files=True):
        self.name = data[0]

        # determine the gender of the speaker on a range, with
        # a default between male and female
        gender_values = {
            "male": -1.0,
            "female": 1.0
        }
        self.gender = gender_values.get(data[6], 0.0)

        self.utterances = {}

        self.create_files = create_files

        self.add_utterance(data[2], data[1])

    def add_utterance(self, text, mp3):

        filename = mp3[:-4]
        self.utterances[text] = filename

        # check to ensure that the .wav file exists in
        #  ../../data/*current_language/wavs/
        if self.create_files:
            dirname = os.path.dirname(data_dir)
            data_dir = os.path.join(dirname, current_language)
            if not os.path.exists(os.path.join(data_dir, "wavs", filename + ".wav")):
                # ensure that the wavs directory exists, and make it if it doesn't
                if not os.path.exists(os.path.join(data_dir, "wavs")):
                    os.mkdir(os.path.join(data_dir, "wavs"))

                # call ffmpeg to convert the mp3 file to a wav file
                subprocess.call(['ffmpeg',
                                 '-loglevel',
                                 'error',
                                 '-i',
                                 os.path.join(data_dir, "clips", mp3),
                                 os.path.join(data_dir, "wavs", filename + ".wav")])

            # check to see if the mfcc file exists in
            # ../../data/*current_language/mfccs/
            if not os.path.exists(os.path.join(data_dir, "mfccs", filename)):
                if not os.path.exists(os.path.join(data_dir, "mfccs")):
                    os.mkdir(os.path.join(data_dir, "mfccs"))

                y, sr = librosa.load(os.path.join(data_dir, "wavs", filename + ".wav"))
                S = librosa.feature.melspectrogram(y=y, sr=sr,
                                                   n_fft=sr * window_length // 1000,
                                                   hop_length=sr * hop_length // 1000)
                mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=mfcc_elems)

                np.save(os.path.join(data_dir, "mfccs", filename + "_mfcc"), mfcc)


def setup_speaker_data(file, create_files=True):
    # start out with an empty speaker dictionary
    speakers = {}

    # Assume that the speech data reference files are in
    # ../../data/*current_language/*file
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "..", "..", "data", current_language, file)

    with open(filename, encoding="utf8") as speaker_file:
        utterance_reader = csv.reader(speaker_file, delimiter='\t')

        # Skip the header in the file
        next(utterance_reader)

        # each line has a new utterance
        for utterance in utterance_reader:
            # if we have already seen this speaker, add the new utterance
            if utterance[0] in speakers:
                speakers[utterance[0]].add_utterance(utterance[2], utterance[1])
            # this is a new speaker, so create them
            else:
                speakers[utterance[0]] = Speaker(utterance, create_files)

    return speakers


def get_encoder_utterance_batches(speakers, num_batches, num_speaker, num_utterance):
    utterance_mfcc_len = (encoder_utterance_length - window_length) // hop_length + 1
    batch = np.zeros((num_speaker * num_utterance,
                      utterance_mfcc_len,
                      mfcc_elems))

    for _ in range(num_batches):
        if len(speakers) < num_speaker:
            raise ValueError("Batch speaker number larger than number of speakers")
        # choose num_speaker random speakers
        speakers_in_batch = random.sample(speakers.keys(), num_speaker)
        # for each speaker
        for i, speaker_name in enumerate(speakers_in_batch):
            speaker = speakers[speaker_name]
            utterances = list(speaker.utterances.values())
            for j in range(num_utterance):
                # find a random utterance, and choose a random utterance_mfcc_len length
                # selection from it
                utterance = random.choice(utterances)
                utterance_mfcc_file = os.path.join(os.path.dirname(__file__),
                                                   "..", "..", "data", current_language, "mfccs",
                                                   utterance + "_mfcc.npy")
                utterance_mfcc = np.load(utterance_mfcc_file)
                utterance_length = utterance_mfcc.shape[1]
                start_index = random.randrange(utterance_length - utterance_mfcc_len)

                batch[i * num_utterance + j] = np.transpose(utterance_mfcc[::,
                                                                           start_index:
                                                                           start_index + utterance_mfcc_len])

        yield batch
