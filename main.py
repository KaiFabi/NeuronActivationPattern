"""Neuron activation pattern visualization.

"""
import argparse
import json
import torch

from pathlib import Path

from src.config import load_config
from src.data import get_dataloader
from src.models import DenseNet, ConvNet
from src.train import train
from src.visualize import visualize


def get_kwargs():
    parser = argparse.ArgumentParser(description="Neuron activation pattern visualization.")
    parser.add_argument(
        "-m", "--mode", required=True, type=str, default="train", choices=("train", "visualize"),
    )
    parser.add_argument(
        "--model", required=True, type=str, default="mlp", choices=("cnn", "mlp"),
    )
    parser.add_argument(
        "--results_dir", default="results"
    )
    parser.add_argument(
        "--weights_dir", default="weights"
    )
    parser.add_argument(
        "--data_dir", default="data"
    )
    parser.add_argument(
        "--runs_dir", default="runs"
    )
    kwargs = parser.parse_args()
    return kwargs


def main():

    kwargs = get_kwargs()

    # Create folder structure
    Path(kwargs.results_dir).mkdir(parents=True, exist_ok=True)
    Path(kwargs.weights_dir).mkdir(parents=True, exist_ok=True)
    Path(kwargs.data_dir).mkdir(parents=True, exist_ok=True)
    Path(kwargs.runs_dir).mkdir(parents=True, exist_ok=True)

    config = load_config(file_path="config.yml")
    config["model"] = kwargs.model
    config["results_dir"] = kwargs.results_dir
    config["weights_dir"] = kwargs.weights_dir
    config["data_dir"] = kwargs.data_dir
    config["runs_dir"] = kwargs.runs_dir
    print(json.dumps(config, indent=4))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    print(f"Using {device}")

    dataloader = get_dataloader(config)

    if kwargs.model == "mlp":
        model = DenseNet(config=config)
    elif kwargs.model == "cnn":
        model = ConvNet(config=config)
    else:
        raise NotImplementedError

    model.to(config["device"])

    if kwargs.mode == "train":
        train(model=model, dataloader=dataloader, config=config)

    elif kwargs.mode == "visualize":
        visualize(model=model, dataloader=dataloader, config=config)


if __name__ == "__main__":
    main()
