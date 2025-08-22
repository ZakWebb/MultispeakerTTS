
from typing import Any, Callable
from lightning.pytorch.core.optimizer import LightningOptimizer
import torch
import torch.nn as nn
import torch.optim as optim

import lightning as pl

import numpy as np

from ..components import PostionalEncoding
from encoder_block import TransformerEncoder


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class TransformerPredictor(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_classes,
        num_heads,
        num_layers,
        lr,
        warmup,
        max_iters,
        dropout=0.0,
        input_dropout=0.0,
    ):
        """TransformerPredictor.
        
        Args:
            input_dim: Hidden dimensionality of the input.
            model_dim: Hidden dimensionality to use inside the transformer
            num_classes: Number of classes to predict per sequence of element
            num_heads: Number of heads to use in the Multi-head Attention blocks
            num_layers: Number of encoder blocks to use
            lr: Learning rate in the optimizer
            warmup: Number of warmup steps.  Usually between 50 and 500
            max_iters: number of maximum iterations the model is trained for.  This is needed
                for the CosineWarmup scheduler
            dropout: Dropout to apply inside the model
            input_dropout: Dropout to apply on the input features
        
        """
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout), # type: ignore
            nn.Linear(self.hparams.input_dim, self.hparams.model_dim), # type: ignore
        )

        # Positional encoding for sequences
        self.positional_encoding = PostionalEncoding(d_model=self.hparams.model_dim) # pyright: ignore[reportAttributeAccessIssue]
        
        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=self.hparams.num_layers, # pyright: ignore[reportAttributeAccessIssue]
            input_dim=self.hparams.model_dim, # pyright: ignore[reportAttributeAccessIssue]
            dim_feedforward=2 * self.hparams.model_dim, # pyright: ignore[reportAttributeAccessIssue]
            num_heads=self.hparams.num_heads, # pyright: ignore[reportAttributeAccessIssue]
            dropout=self.hparams.dropout, # pyright: ignore[reportAttributeAccessIssue]
        )

        # Output classifier per sequence element
        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim), # pyright: ignore[reportAttributeAccessIssue]
            nn.LayerNorm(self.hparams.model_dim), # pyright: ignore[reportAttributeAccessIssue]
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout), # pyright: ignore[reportAttributeAccessIssue]
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes), # pyright: ignore[reportAttributeAccessIssue]
        )

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Args:
            x: Input features of shape [Batch, Seqlen, input_dim]
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If true, we add the positional encoding to the input
        
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)

        return x
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr) # pyright: ignore[reportAttributeAccessIssue]

        # We don't return the lr scheduler becasue we need to apply it per iteration, not per epoch
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_iters) # pyright: ignore[reportAttributeAccessIssue]
        
        return optimizer
    
    def optimizer_step(self, epoch: int, batch_idx: int, optimizer: optim.Optimizer | LightningOptimizer, optimizer_closure: Callable[[], Any] | None = None) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.lr_scheduler.step()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def validation_step(self, *args: Any, **kwargs: Any):
        raise NotImplementedError
    
    def test_step(self, batch, batch_idx):
        raise NotImplementedError

