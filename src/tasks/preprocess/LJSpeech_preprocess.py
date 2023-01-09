from tasks.preprocess.base_preprocess_task import BasePreprocessTask, register_preprocessor

import os
import random
import json

@register_preprocessor
class LJSpeechPreprocess(BasePreprocessTask):
    def __init__(self, config):
        super(LJSpeechPreprocess, self).__init__(config)

        self.texts = self.get_text()

    def get_text(self):
        texts = []
        with open(os.path.join(self.raw_data_folder, "metadata.csv"), "r") as f:
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
            for id, text in texts:
                current_ids.append(id)
                with open(os.path.join(self.data_folder, type, "raw_text", id + ".txt"), "w") as f:
                    f.write(text)
                cleaned_text = self.cleaner.convert(text)
                with open(os.path.join(self.data_folder, type, "cleaned_text", id + ".txt"), "w") as f:
                    f.write(cleaned_text)
                phonemes = self.t2p.convert(cleaned_text)
                with open(os.path.join(self.data_folder, type, "phonemes", id + ".json"), "w") as f:
                    json.dump(phonemes, f, indent=2)
                self.audio.load_wav(os.path.join(self.raw_data_folder, "wavs"), id)
                self.audio.process_wav()
                self.audio.save_wav(os.path.join(self.data_folder, type, "wavs"), id)
                self.audio.save_mel(os.path.join(self.data_folder, type, "mels"), id)
            
            with open(os.path.join(self.data_folder, type, type + "_metadata.csv"), "w") as f:
                for id in current_ids:
                    f.write(id + "\n")


