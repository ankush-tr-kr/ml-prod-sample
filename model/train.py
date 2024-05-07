"""
This module defines a function to train a CNN model using the network architecture in the
network module and the data loaders from the preprocess module. The hyperparameters are optimized
using Optuna.
"""

from datetime import datetime

import optuna
import torch
from torch import nn, optim

from .network import MNISTClassifier
from .preprocess import get_data_loaders
from .train_utils import train_model


# pylint: disable=too-many-locals
# pylint: disable=no-member
def objective(trial):
    """
    Objective function for Optuna to optimize. This function is called by Optuna with different
    hyperparameter values and returns the validation accuracy of the model trained with those
    hyperparameters.

    Args:
        trial (optuna.Trial): Optuna object that stores the hyperparameters to be optimized.

    Returns:
        float: Validation accuracy of the model trained with the hyperparameters.

    """

    # setting the device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters to be optimized
    batch_size = trial.suggest_int("batch_size", 32, 256)

    # Load data using the imported function
    train_loader, val_loader = get_data_loaders(batch_size=batch_size)

    # Initialize model
    model = MNISTClassifier().to(device)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    learn_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learn_rate)

    # Training loop
    criterion = (
        nn.CrossEntropyLoss()
    )  # using CrossEntropyLoss for multi-class classification

    num_epochs = 3  # Keep it small for hyperparameter optimization

    train_model(model, train_loader, optimizer, criterion, device, num_epochs)

    # Validation accuracy calculation
    correct = 0
    total = 0
    with torch.no_grad():
        for (
            data,
            target,
        ) in val_loader:  # Using validation loader for accuracy calculation
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output.data, 1)
            total += target.size(0)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / total
    # print(trial.report(accuracy, epoch))

    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return accuracy
