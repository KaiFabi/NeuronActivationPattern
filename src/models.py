from math import prod

import torch
import torch.nn as nn


class DenseResNet(nn.Module):
    """Fully connected neural network with residual connections.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.n_dims_in = prod(config["input_shape"])
        self.n_dims_out = config["n_classes"]
        self.n_dims_hidden = config["n_dims_hidden"]
        self.classifier = self.make_classifier(n_blocks=config["n_blocks"])
        self.weights_init()

        # Dictionary to store the activations
        self.activations = {}

    def make_classifier(self, n_blocks: int) -> torch.nn.Module:

        layers = []

        # Input layer
        layers += [
            nn.Linear(in_features=self.n_dims_in, out_features=self.n_dims_hidden),
            nn.BatchNorm1d(num_features=self.n_dims_hidden),
            nn.ReLU(),
        ]

        # Hidden layers
        for _ in range(n_blocks):
            layers += [Block(num_features=self.n_dims_hidden), ]

        # Output layer
        layers += [nn.Linear(in_features=self.n_dims_hidden, out_features=self.n_dims_out),]

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.n_dims_in)
        x = self.classifier(x)
        return x

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook

    def weights_init(self) -> None:
        for module in self.children():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight.data, nonlinearity="relu")
                torch.nn.init.zeros_(module.bias.data)


class Block(nn.Module):
    """Fully connected block with residual connection."""

    def __init__(self, num_features: int):
        super().__init__()

        self.linear1 = nn.Linear(in_features=num_features, out_features=num_features)
        self.linear2 = nn.Linear(in_features=num_features, out_features=num_features)

        self.bn1 = nn.BatchNorm1d(num_features=num_features)
        self.bn2 = nn.BatchNorm1d(num_features=num_features)

        self.af1 = torch.nn.ReLU()
        self.af2 = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.bn1(x)
        out = self.af1(out)
        out = self.linear1(out)

        out = self.bn2(out)
        out = self.af2(out)
        out = self.linear2(out)

        out = out + identity

        return out
