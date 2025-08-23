import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L

from .modules.vocoders import WaveGlow
from .data_loader import LJSpeech11


my_model = WaveGlow()

lightning_data = LJSpeech11("/mnt/c/Users/zakww/Documents/Speech Data/LJSpeech-1.1/",
                            compute_mel_spectrogram=True)


dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(my_model, datamodule=lightning_data)