from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import os
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import pandas as pd
import numpy as np
import librosa
from functools import partial

VALID_DATA_TYPES_LJSPEECH = ["mel_spectrogram", "text", "cleaned_text", "wav_file"]
RANDOM_SEED = 42


class LJSpeech11Data(Dataset):
    def __init__(self, 
                 data_dir = "./", 
                 input_data_type: str = "oops", 
                 output_data_type: str = "oops",
        ):
        super().__init__()
        self.input_data_type = input_data_type
        self.output_data_type = output_data_type
        self.data_dir = data_dir
        self.wav_dir = data_dir + "wavs/"
        self.mel_dir = data_dir + "mels/"
        self.texts = pd.read_csv(data_dir + "metadata.csv", sep='|', header=None)
        self.valid_types = [input_data_type, output_data_type]

        self.num_utterances=13100

    def __len__(self):
        if self.input_data_type in VALID_DATA_TYPES_LJSPEECH and self.input_data_type in VALID_DATA_TYPES_LJSPEECH:
            return self.num_utterances
        return 0

    def __getitem__(self, index):
        in_item = None
        out_item = None

        data_type_check = "mel_spectrogram"
        if self.input_data_type == data_type_check or self.output_data_type == data_type_check:
            name = self.texts.iat[index, 0]
            data = np.load(self.mel_dir + name + ".mel.npy")

            if self.input_data_type == data_type_check:
                in_item = data
            if self.output_data_type == data_type_check:
                out_item = data
        
        
        data_type_check = "text"
        if self.input_data_type == data_type_check or self.output_data_type == data_type_check:
            data = self.texts.iat[index, 1]
            if self.input_data_type == data_type_check:
                in_item = data
            if self.output_data_type == data_type_check:
                out_item = data
        
        
        data_type_check = "cleaned_text"
        if self.input_data_type == data_type_check or self.output_data_type == data_type_check:
            data = self.texts.iat[index, 2]
            if self.input_data_type == data_type_check:
                in_item = data
            if self.output_data_type == data_type_check:
                out_item = data
        
        
        data_type_check = "wav_file"
        if self.input_data_type == data_type_check or self.output_data_type == data_type_check:
            name = self.texts.iat[index, 0]
            data, _ = librosa.load(self.wav_dir  + name + ".wav")
            if self.input_data_type == data_type_check:
                in_item = data
            if self.output_data_type == data_type_check:
                out_item = data

        return in_item, out_item



class LJSpeechDataModule(LightningDataModule):
    """`LightningDataModule` for the LJSpeech dataset

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (11_600, 500, 1_000),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        input_data_type: str = "mel_spectrogram",
        output_data_type: str = "wav_file",
        njt: bool = False,
    ) -> None:
        """Initialize a `LJSpeechDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `32`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose([]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

#   I don't know how to download the LJSpeech Dataset yet.  Eventually I'll get there
#
#
#    def prepare_data(self) -> None:
#        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
#        within a single process on CPU, so you can safely add your downloading logic within. In
#        case of multi-node training, the execution of this hook depends upon
#        `self.prepare_data_per_node()`.
#
#        Do not use it to assign state (self.x = y).
#        """
#        MNIST(self.hparams.data_dir, train=True, download=True)
#        MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = LJSpeech11Data(self.hparams.data_dir, self.hparams.input_data_type, self.hparams.output_data_type)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(RANDOM_SEED),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=partial(_collate_fn_for_NJT_Tensors, njt=self.hparams.njt),
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=partial(_collate_fn_for_NJT_Tensors, njt=self.hparams.njt),
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=partial(_collate_fn_for_NJT_Tensors, njt=self.hparams.njt),
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = LJSpeechDataModule()


def _collate_fn_for_NJT_Tensors(batch, njt=False):
    sequences = [a for a,_ in batch]
    labels = [b for _,b in batch]

    sequences_NJT = torch.nested.nested_tensor(sequences, layout=torch.jagged)
    labels_NJT = torch.nested.nested_tensor(labels, layout=torch.jagged)

    if not njt:
        sequences_NJT = torch.nested.to_padded_tensor(sequences_NJT, 0)
        labels_NJT = torch.nested.to_padded_tensor(labels_NJT, 0)

    return sequences_NJT, labels_NJT