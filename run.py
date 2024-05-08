"""
this file is used to run the optuna study
"""

from datetime import datetime

import optuna
import torch
from model.train import get_data_loaders, MNISTClassifier, objective
from model.train_utils import train_model
from torch import nn, optim


def retrain_best_model(study):
    """
    retrains the model with the best hyperparameters
    """
    # Get the best trial
    best_trial = study.best_trial

    # Print the best trial's details
    print("Starting retraining...")
    print(f"Value: {best_trial.value}")

    # Hyperparameters
    best_params = best_trial.params
    batch_size = best_params["batch_size"]
    learn_rate = best_params["lr"]
    optimizer_name = best_params["optimizer"]

    # set device to CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data using the imported function
    train_loader, val_loader = get_data_loaders(batch_size=batch_size)

    # Create and train the model using the best hyperparameters
    model = MNISTClassifier().to(device)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learn_rate)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    num_epochs = 50
    train_model(model, train_loader, optimizer, criterion, device, num_epochs)

    # define model name using datetime and path to save the model
    model_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"saved_models/{model_name}.pth"

    # Save the model locally
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    # json file for the model that contains information about the model
    # and the hyperparameters used
    model_info = {
        "model_name": model_name,
        "model_path": model_path,
        "hyperparameters": best_params,
    }
    model_info_path = f"saved_models/{model_name}.json"
    torch.save(model_info, model_info_path)
    print(f"Model info saved at {model_info_path}")


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",
        study_name=f"{datetime.now()}-mnist-example",
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=5)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    retrain_best_model(study)
