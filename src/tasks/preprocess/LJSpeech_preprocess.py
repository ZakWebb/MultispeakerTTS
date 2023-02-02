from tasks.preprocess.base_preprocess_task import BasePreprocessTask, register_preprocessor

import os
import random
import json
import tqdm

@register_preprocessor
class LJSpeechPreprocess(BasePreprocessTask):
    def __init__(self, config):
        super(LJSpeechPreprocess, self).__init__(config)

        self.texts = self.get_text()

    def get_text(self):
        train_metadata_exists = os.path.exists(os.path.join(self.data_dir, "train")) and os.path.exists(os.path.join(self.data_dir, "train", "train_metadata.csv"))
        valid_metadata_exists = os.path.exists(os.path.join(self.data_dir, "valid")) and os.path.exists(os.path.join(self.data_dir, "valid", "valid_metadata.csv"))
        test_metadata_exists = os.path.exists(os.path.join(self.data_dir, "test")) and os.path.exists(os.path.join(self.data_dir, "test", "test_metadata.csv"))

        if train_metadata_exists and valid_metadata_exists and test_metadata_exists:
            returnable = {}
            for split in {"train", "valid", "test"}:
                texts = []
                with open(os.path.join(self.data_dir, split, "{}_metadata.csv".format(split))) as f:
                    for line in f:
                        line = line.strip()
                        with open(os.path.join(self.data_dir, split, "data", "{}.txt".format(line))) as g:
                            text = g.readline().strip()
                            texts.append((line, text))
                returnable[split] = texts
            return returnable

        else:
            texts = []
            with open(os.path.join(self.raw_data_dir, "metadata.csv"), "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("|")
                    texts.append((parts[0], parts[1]))
        


            random.shuffle(texts)
            length = len(texts)
            return {"train" : texts[:round(self.train_percentage * length)], \
                    "valid" : texts[round(self.train_percentage * length): round((self.train_percentage + self.valid_percentage) * length)], \
                    "test" : texts[round((self.train_percentage + self.valid_percentage) * length):]}
        
    def build_files(self):
        for type in {"train", "valid", "test"}:
            texts = self.texts[type]
            current_ids = []
            for id, text in tqdm.tqdm(texts, desc="Working on {} inputs".format(type)):
                current_ids.append(id)
                raw_text_exists = os.path.exists(os.path.join(self.data_dir, type, "raw_text", id + ".txt"))
                cleaned_text_exists = os.path.exists(os.path.join(self.data_dir, type, "data", id + ".txt"))
                phonemes_exist = os.path.exists(os.path.join(self.data_dir, type, "phonemes", id + ".json"))
                audio_exists = os.path.exists(os.path.join(self.data_dir, type, "data", id + ".wav"))
                mel_exists = os.path.exists(os.path.join(self.data_dir, type, "mels", id + ".mel"))


                if not raw_text_exists:
                    with open(os.path.join(self.data_dir, type, "raw_text", id + ".txt"), "w") as f:
                        f.write(text)

                if not cleaned_text_exists or not phonemes_exist:
                    cleaned_text = self.cleaner.convert(text)
                if not cleaned_text_exists:
                    with open(os.path.join(self.data_dir, type, "data", id + ".txt"), "w") as f:
                        f.write(cleaned_text)
                # if not phonemes_exist:
                #     phonemes = self.t2p.convert(cleaned_text)
                #     with open(os.path.join(self.data_dir, type, "phonemes", id + ".json"), "w") as f:
                #         json.dump(phonemes, f, indent=2)

                if not audio_exists or not mel_exists:
                    self.audio.load_wav(os.path.join(self.raw_data_dir, "wavs"), id)
                    self.audio.process_wav()
                if not audio_exists:
                    self.audio.save_wav(os.path.join(self.data_dir, type, "data"), id)
                if not mel_exists:
                    self.audio.save_mel(os.path.join(self.data_dir, type, "mels"), id)
            
            if not os.path.exists(os.path.join(self.data_dir, type, type + "_metadata.csv")):
                with open(os.path.join(self.data_dir, type, type + "_metadata.csv"), "w") as f:
                    for id in current_ids:
                        f.write(id + "\n")
    
    def process_textgrids(self):
        pass
        


