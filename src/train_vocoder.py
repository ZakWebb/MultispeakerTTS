import os
from torch import optim, nn, utils, Tensor
import lightning as L

from modules.vocoders import WaveGlowLightning
from data_loader import LJSpeech11

my_model = WaveGlowLightning()

# Figure out a better way to do this with config files

# Laptop location: "/mnt/c/Users/zakww/Documents/Speech Data/LJSpeech-1.1/"
# Desktop location: "/mnt/d/Speech Data/LJ/LJSpeech-1.1/"

lightning_data = LJSpeech11("/mnt/d/Speech Data/LJ/LJSpeech-1.1/",
                            compute_mel_spectrogram=True)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(my_model, datamodule=lightning_data)