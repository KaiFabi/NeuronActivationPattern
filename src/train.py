"""Script holds method for neural network training.
"""
import datetime
import os

from tqdm import tqdm

import torch
import torch.optim as optim
from torch import nn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from .stats import comp_loss_accuracy


def train(model: torch.nn.Module, dataloader: tuple[DataLoader, DataLoader], config: dict) -> None:
    """

    Args:
        model: PyTorch model.
        dataloader: Tuple holding training and test dataloader.
        config: Dictionary holding configuration for training.

    """
    uid = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')

    writer = SummaryWriter(log_dir=f"{config['runs_dir']}/{uid}")
    run_training(model=model, dataloader=dataloader, writer=writer, config=config)
    writer.close()


def run_training(model: torch.nn.Module,
                 dataloader: tuple[DataLoader, DataLoader],
                 writer: SummaryWriter,
                 config: dict) -> None:
    """Main training logic.

    Args:
        model: PyTorch model.
        dataloader: Training and test data loader.
        writer: Tensorboard writer instance.
        config: Dictionary holding configuration for training.

    """
    model_name = config["model"]
    device = config["device"]
    dataset = config["dataset"]
    n_epochs = config["n_epochs"]
    save_train_stats_every_n_epochs = config["save_train_stats_every_n_epochs"]
    save_test_stats_every_n_epochs = config["save_test_stats_every_n_epochs"]
    save_model_every_n_epochs = config["save_model_every_n_epochs"]

    trainloader, testloader = dataloader

    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(n_epochs)):

        running_loss = 0.0
        running_accuracy = 0.0
        running_counter = 0

        model.train()
        for x_data, y_data in trainloader:

            # get the inputs; data is a list of [inputs, lables]
            inputs, labels = x_data.to(device), y_data.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + gradient descent
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # keeping track of statistics
            running_loss += loss.item()
            running_accuracy += (torch.argmax(outputs, dim=1) == labels).float().sum()
            running_counter += labels.size(0)

        running_loss = running_loss / running_counter
        running_accuracy = running_accuracy / running_counter

        if epoch % save_train_stats_every_n_epochs == 0:
            writer.add_scalar("train_loss", running_loss, epoch)
            writer.add_scalar("train_accuracy", running_accuracy, epoch)

        if epoch % save_test_stats_every_n_epochs == 0:
            test_loss, test_accuracy = comp_loss_accuracy(model=model, criterion=criterion,
                                                          data_loader=testloader,
                                                          device=device)
            writer.add_scalar("test_loss", test_loss, epoch)
            writer.add_scalar("test_accuracy", test_accuracy, epoch)

        if epoch % save_model_every_n_epochs == 0:
            weights_path = os.path.join(config["weights_dir"], f"{dataset}_{model_name}.pth")
            torch.save(model.state_dict(), weights_path)

        print(f"{epoch:04d} {running_loss:.5f} {running_accuracy:.4f}")
