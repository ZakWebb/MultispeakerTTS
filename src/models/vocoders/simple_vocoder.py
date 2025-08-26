import torch
from torch import nn

class SimpleVocoder(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 80,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 1,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, mels, series = x.size()

        # (batch, mels, series) -> (batch, series, mels)
        x = torch.transpose(x, -2, -1)

        # Modify mels
        x = self.model(x)

        # (batch, series, output_size) -> (batch, series * output_size)
        x = torch.transpose(x, -2, -1)

        x = torch.flatten(x, -2, -1)

        return x


if __name__ == "__main__":
    _ = SimpleVocoder()