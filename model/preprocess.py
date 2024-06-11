"""
This module defines a function to get data loaders for the MNIST dataset,
which facilitate loading and batching the data during training and
validation of a CNN model.
"""

import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(batch_size=64, train=True, val=True):
    """Function to get data loaders for the MNIST dataset.

    Args:
        batch_size (int, optional): Batch size for the data loader.
        Default 64.
        train (bool, optional): Whether to get the training set data loader.
        Default True.
        val (bool, optional): Whether to get the validation set data loader.
        Default True.

    Returns:
        tuple: Data loaders for the training and validation sets.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]  # mean and std of MNIST dataset
    )

    # NOTE: we need to implement a hash check instead of only checking 
    # if the file exists
    mnist_train_data_path = "./data/MNIST/raw/train-images-idx3-ubyte.gz"
    mnist_val_data_path = "./data/MNIST/raw/t10k-images-idx3-ubyte.gz"

    # Download the dataset if it doesn't exist

    train_dataset = (
        datasets.MNIST(
            "./data",
            train=True,
            download=not os.path.exists(mnist_train_data_path),
            transform=transform,
        )
        if train
        else None
    )
    val_dataset = (
        datasets.MNIST(
            "./data",
            train=False,
            download=not os.path.exists(mnist_val_data_path),
            transform=transform,
        )
        if val
        else None
    )

    train_loader = (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if train
        else None
    )
    val_loader = (
        DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False) if val else None
    )

    return train_loader, val_loader


if __name__ == "__main__":
    get_data_loaders()
