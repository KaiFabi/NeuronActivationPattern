from math import prod

import torch
import torch.nn as nn


class DenseNet(nn.Module):
    """Fully connected neural network with residual connections.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.n_dims_in = prod(config["input_shape"])
        self.n_dims_hidden = config["n_dims_hidden"]
        self.n_dims_out = config["n_classes"]
        self.classifier = self._get_dense_net(n_blocks=config["n_dense_blocks"])
        self.weights_init()

        # Dictionary to store the activations
        self.activations = {}

    def _get_dense_net(self, n_blocks: int) -> torch.nn.Module:

        layers = []

        # Input layer
        layers += [
            nn.Linear(in_features=self.n_dims_in, out_features=self.n_dims_hidden),
            nn.BatchNorm1d(num_features=self.n_dims_hidden),
            nn.ReLU(),
        ]

        # Hidden layers
        for _ in range(n_blocks):
            layers += [DenseBlock(num_features=self.n_dims_hidden), ]

        # Output layer
        layers += [nn.Linear(in_features=self.n_dims_hidden, out_features=self.n_dims_out),]

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.n_dims_in)
        x = self.classifier(x)
        return x

    # -------------------------------------------------------- Activation Pattern --------

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook

    def weights_init(self) -> None:
        for module in self.children():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight.data, nonlinearity="relu")
                if module.bias.data is not None:
                    torch.nn.init.zeros_(module.bias.data)


class ConvNet(nn.Module):
    """Convolutional neural network with residual connections.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.n_channels_in = config["input_shape"][-1]
        self.n_channels_hidden = config["n_channels_hidden"]
        self.n_channels_out = config["n_channels_out"]
        self.n_dims_out = config["n_classes"]
        self.in_features_dense = config["n_channels_out"] * (config["input_shape"][0]//2)**2

        self.conv_net = self._get_conv_net(n_blocks=config["n_conv_blocks"])
        self.dense_net = self._get_dense_net(n_blocks=config["n_dense_blocks"])
        self.weights_init()

        # Dictionary to store the activations
        self.activations = {}

    def _get_conv_net(self, n_blocks: int) -> torch.nn.Module:

        layers = []

        # Input layer
        layers += [
                nn.Conv2d(in_channels=self.n_channels_in, out_channels=self.n_channels_hidden,
                          kernel_size=(2, 2), stride=(2, 2)),
                nn.BatchNorm2d(num_features=self.n_channels_hidden),
                torch.nn.ReLU()
                ]

        # Hidden layers
        for _ in range(n_blocks):
            layers += [ConvBlock(num_channels=self.n_channels_hidden)]

        # Output layer
        layers += [
                nn.Conv2d(in_channels=self.n_channels_hidden, out_channels=self.n_channels_out,
                          kernel_size=(3, 3), padding="same"),
                nn.BatchNorm2d(num_features=self.n_channels_out),
                torch.nn.ReLU()
                ]

        return nn.Sequential(*layers)

    def _get_dense_net(self, n_blocks: int) -> torch.nn.Module:

        return nn.Linear(in_features=self.in_features_dense, out_features=self.n_dims_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_net(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense_net(x)
        return x

    # -------------------------------------------------------- Activation Pattern --------

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook

    def weights_init(self) -> None:
        for module in self.children():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight.data, nonlinearity="relu")
                if module.bias.data is not None:
                    torch.nn.init.zeros_(module.bias.data)


class DenseBlock(nn.Module):
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


class ConvBlock(nn.Module):
    """Convolutional block with residual connection."""

    def __init__(self, num_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=num_channels,
                               out_channels=num_channels,
                               kernel_size=(3, 3), padding="same")
        self.conv2 = nn.Conv2d(in_channels=num_channels,
                               out_channels=num_channels,
                               kernel_size=(3, 3), padding="same")

        self.bn1 = nn.BatchNorm2d(num_features=num_channels)
        self.bn2 = nn.BatchNorm2d(num_features=num_channels)

        self.af1 = torch.nn.ReLU()
        self.af2 = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.bn1(x)
        out = self.af1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.af2(out)
        out = self.conv2(out)

        out = out + identity

        return out
