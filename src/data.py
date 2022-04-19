"""Script creates data loaders for different datasets.
"""
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloader(config: dict) -> tuple[DataLoader, DataLoader]:

    dataset = config["dataset"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    data_dir = config["data_dir"]

    if dataset == "cifar10":

        avg = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        train_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(avg, std)
        ]

        test_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(avg, std)
        ]

        transform_train = transforms.Compose(train_transforms)
        transform_test = transforms.Compose(test_transforms)

        trainset_config = dict(root=data_dir, train=True, download=True, transform=transform_train)
        trainset = torchvision.datasets.CIFAR10(**trainset_config)

        trainloader_config = dict(dataset=trainset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=True)
        trainloader = DataLoader(**trainloader_config)

        testset_config = dict(root=data_dir, train=False, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR10(**testset_config)

        testloader_config = dict(dataset=testset, batch_size=2*batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=True)
        testloader = DataLoader(**testloader_config)

    elif dataset == "fmnist":

        # Fashion-MNIST
        avg = (0.2859, )
        std = (0.3530, )

        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=45),
                transforms.RandomCrop(28, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(avg, std)
            ]
        )

        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(avg, std)]
        )

        trainset = torchvision.datasets.FashionMNIST(root=data_dir,
                                                     train=True,
                                                     download=True,
                                                     transform=transform_train)

        trainloader = DataLoader(trainset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=num_workers)

        testset = torchvision.datasets.FashionMNIST(root=data_dir,
                                                    train=False,
                                                    download=True,
                                                    transform=transform_test)

        testloader = DataLoader(testset,
                                batch_size=2*batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=num_workers)

    elif dataset == "mnist":

        # Fashion-MNIST
        avg = (0.1307, )
        std = (0.3081, )

        transform_train = transforms.Compose(
            [
                # transforms.RandomRotation(degrees=45),
                # transforms.RandomCrop(28, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(avg, std)
            ]
        )

        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(avg, std)]
        )

        trainset = torchvision.datasets.MNIST(root=data_dir,
                                                     train=True,
                                                     download=True,
                                                     transform=transform_train)

        trainloader = DataLoader(trainset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=num_workers)

        testset = torchvision.datasets.MNIST(root=data_dir,
                                                    train=False,
                                                    download=True,
                                                    transform=transform_test)

        testloader = DataLoader(testset,
                                batch_size=2*batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=num_workers)

    else:
        raise NotImplementedError(f"No dataloader for dataset {dataset} implemented.")

    return trainloader, testloader
