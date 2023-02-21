from tasks.preprocess.base_preprocess_task import BasePreprocessTask, register_preprocessor
from data_gen.symbols import symbol_to_id

import numpy as np
import os
import random
import json
import tqdm
import textgrids
import torch

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
                energy_frame_exists = os.path.exists(os.path.join(self.data_dir, type, "energy", id + "_frame.energy"))
                f0_frame_exists = os.path.exists(os.path.join(self.data_dir, type, "f0", id + "_frame.f0"))


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

                if not audio_exists or not mel_exists or not energy_frame_exists or not f0_frame_exists:
                    self.audio.load_wav(os.path.join(self.raw_data_dir, "wavs"), id)
                    self.audio.process_wav()
                if not audio_exists:
                    self.audio.save_wav(os.path.join(self.data_dir, type, "data"), id)
                if not mel_exists:
                    self.audio.save_mel(os.path.join(self.data_dir, type, "mels"), id)
                if not energy_frame_exists:
                    self.audio.save_energy(os.path.join(self.data_dir, type, "energy"), id + "_frame")
                if not f0_frame_exists:
                    self.audio.save_f0(os.path.join(self.data_dir, type, "f0"), id + "_frame")
            
            if not os.path.exists(os.path.join(self.data_dir, type, type + "_metadata.csv")):
                with open(os.path.join(self.data_dir, type, type + "_metadata.csv"), "w") as f:
                    for id in current_ids:
                        f.write(id + "\n")
    
    def check_dir_complete(self, split, datatype):
        texts = self.texts[split]
        
        if datatype in {"wavs", "cleaned_text"}:
            datadir = "data"
        else:
            datadir = datatype

        extensions = {
            "cleaned_text": ".txt",
            "raw_text": ".txt",
            "wavs": ".wav",
            "mels": ".mel",
            "textgrids": ".TextGrid",
            "durations": ".dur",
            "phonemes": ".json"
        }

        extension = extensions[datatype]

        for id, _ in texts:
            if not os.path.exists(os.path.join(self.data_dir, split, datadir, id + extension)):
                print("missing {} file".format(id + extension))
                return False
        
        return True

    def process_textgrids(self):
        for datatype in {"train", "valid", "test"}:
            texts = self.texts[datatype]

            for id, _ in tqdm.tqdm(texts, desc="Working on {} textgrids".format(datatype)):
                grid = textgrids.TextGrid(os.path.join(self.data_dir, datatype, "textgrids", id + ".TextGrid"))
                phones = list(map(lambda x: x['label'], grid.interval_tier_to_array("phones")))
                phone_ids = list(map(lambda x: symbol_to_id["@sp" if x == '' else "@" + x], phones))
                dur = list(map(lambda x: round(self.sample_rate * (x['end'] - x['begin']) / self.hop_length), grid.interval_tier_to_array("phones")))
                dur = np.array(dur)

                with open(os.path.join(self.data_dir, datatype, "phonemes", id + ".json"),'w') as f:
                    json.dump(phone_ids, f)                
                
                torch.save(torch.from_numpy(dur), os.path.join(self.data_dir, datatype, "durations", id + ".dur"))

                # compute energy for each phone
                energy_frame_filename = os.path.join(self.data_dir, datatype, "energy", id + "_frame.energy")
                if os.path.exists(os.path.exists(energy_frame_filename)):
                    energy_frame = torch.load(energy_frame_filename)
                    energy_phone = self.average_by_duration(energy_frame.detach().numpy(), dur)
                    torch.save(torch.tensor(energy_phone), os.path.join(self.data_dir, datatype, "energy", id + "_phone.energy"))
                
                # compute f0 for each phone
                f0_frame_filename = os.path.join(self.data_dir, datatype, "f0", id + "_frame.f0")
                if os.path.exists(os.path.exists(f0_frame_filename)):
                    f0_frame = torch.load(f0_frame_filename)
                    f0_phone = self.average_by_duration(f0_frame.detach().numpy(), dur)
                    torch.save(torch.tensor(f0_phone), os.path.join(self.data_dir, datatype, "f0", id + "_phone.f0"))
