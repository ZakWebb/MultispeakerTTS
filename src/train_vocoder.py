
import lightning as L

from modules.vocoders import SimpleVocoder
from data_loader import LJSpeech11

from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="./conf/", config_name="config")
def get_config(cfg: DictConfig) -> DictConfig:
    return cfg

if __name__ == "__main__":
    config = get_config()


    my_model = SimpleVocoder()

# Figure out a better way to do this with config files

# Laptop location: "/mnt/c/Users/zakww/Documents/Speech Data/LJSpeech-1.1/"
# Desktop location: "/mnt/d/Speech Data/LJ/LJSpeech-1.1/"

    lightning_data = LJSpeech11(config)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = L.Trainer(config)
    trainer.fit(my_model, datamodule=lightning_data)