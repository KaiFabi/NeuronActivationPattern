"""Script visualizes neuron activation pattern of dense ResNet.
"""
import matplotlib.pyplot as plt
import os
import torch

from torch.utils.data import DataLoader

from .models import DenseBlock, ConvBlock


def visualize(model: torch.nn.Module, dataloader: tuple, config: dict) -> None:
    """Method visualizes average activation patterns per class for provided network.

    Args:
        model: PyTorch model.
        dataloader: Tuple holding training and test dataloader.
        config: Dictionary holding configuration for training.

    """
    # Load pretrained network
    model = load_checkpoint(model=model, config=config)

    # Register forward hooks
    if config["model"] == "mlp":
        register_forward_hooks(model=model, module=DenseBlock)
    elif config["model"] == "cnn":
        register_forward_hooks(model=model, module=ConvBlock)
    else:
        raise NotImplementedError

    # Get dataloader
    _, testloader = dataloader

    # Extract activation pattern
    class_activations = get_class_activations(model=model, dataloader=testloader, config=config)

    # Plot pattern
    plot_class_activations(class_activations=class_activations, config=config)


def load_checkpoint(model: torch.nn.Module, config: dict):
    """Loads checkpoint of pretrained model.

    Args:
        model: PyTorch model.
        config: Dictionary holding configuration.

    Returns:
        Pytorch model.

    """
    weights_dir = config["weights_dir"]
    dataset = config["dataset"]
    model_name = config["model"]
    device = config["device"]

    model.load_state_dict(torch.load(os.path.join(weights_dir, f"{dataset}_{model_name}.pth")))
    model.to(device)

    return model


def register_forward_hooks(model: torch.nn.Module, module) -> None:
    """Registers forward hooks in model.

    Module can be any module operations such as DenseBlock, ConvBlock,
    nn.BatchNorm1d, nn.Linear, nn.ReLU. Modules should have same output
    dimensions.

    Args:
        model: PyTorch model.
        module: Module of neural network.

    """
    for name, layer in model.named_modules():
        if isinstance(layer, module):
            layer.register_forward_hook(model.get_activation(name))


def get_class_activations(model: torch.nn.Module, dataloader: DataLoader, config: dict) -> dict:
    """Collect neuron activation patterns for each class.

    Args:
        model: PyTorch model.
        dataloader: PyTorch dataloader.
        config: Dictionary holding configuration.

    Returns:
        Dictionary holding activation patterns for each data sample separated by class.

    """
    device = config["device"]
    class_activations = {class_: [] for class_ in dataloader.dataset.classes}
    idx_to_class = {value: key for key, value in dataloader.dataset.class_to_idx.items()}
    model.eval()

    for i, (x_data, y_data) in enumerate(dataloader):

        # Infer data
        inputs, labels = x_data.to(device), y_data.to(device)
        logits = model(inputs)
        preds = torch.argmax(logits, dim=-1)

        # Convert activations to a single tensor
        activation_pattern = [activation.flatten(1) for activation in model.activations.values()]
        activation_pattern = torch.stack(activation_pattern, dim=-1).detach().cpu()

        # Assign activation pattern to each class.
        for j, pattern in enumerate(activation_pattern):
            if preds[j] == labels[j]:
                idx = y_data[j].item()
                class_activations[idx_to_class[idx]].append(pattern)

    for key in class_activations:
        class_activations[key] = torch.stack(class_activations[key], dim=-1)

    return class_activations


def plot_class_activations(class_activations: dict, config: dir) -> None:
    """Visualizes activations patterns per class.

    Args:
        class_activations: Dictionary with activations per class.
        config: Dictionary holding configuration.

    """
    n_classes = config["n_classes"]
    dataset = config["dataset"]
    model_name = config["model"]

    plt_cfg = {"nrows": n_classes, "ncols": 1, "figsize": (8, 8)}
    img_cfg = {"extent": (0, 4000, 0, 400), "interpolation": "nearest", "cmap": "rainbow"}
    save_cfg = {"dpi": 240, "bbox_inches": "tight", "transparent": True, "format": "png"}

    # Mean activation
    fig, axes = plt.subplots(**plt_cfg)
    for i, ((name, activations), ax) in enumerate(zip(class_activations.items(), axes.flatten())):
        im = ax.matshow(activations.mean(dim=-1), **img_cfg)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(name.capitalize(), fontsize=6)
    fig.colorbar(im, ax=axes.ravel().tolist())
    file_name = f"{dataset}_{model_name}_mean_activation_pattern.png"
    results_path = os.path.join(config["results_dir"], file_name)
    plt.savefig(results_path, **save_cfg)
    plt.close(fig)

    # Standard deviation
    fig, axes = plt.subplots(**plt_cfg)
    for (name, activations), ax in zip(class_activations.items(), axes.flatten()):
        im = ax.matshow(activations.std(dim=-1), **img_cfg)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(name.capitalize(), fontsize=6)
    fig.colorbar(im, ax=axes.ravel().tolist())
    file_name = f"{dataset}_{model_name}_std_activation_pattern.png"
    results_path = os.path.join(config["results_dir"], file_name)
    plt.savefig(results_path, **save_cfg)
    plt.close(fig)

