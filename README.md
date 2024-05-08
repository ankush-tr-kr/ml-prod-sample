# MNIST Digits Classification

This project offers a solution for classifying MNIST handwritten digits using a Convolutional Neural Network (CNN) and PyTorch. 

## Setup

Clone the project repository  & create a virtual environment and install the required packages using Pip

```
pip install -r requirements.txt
```
Alternatively conda can also be used

```
conda env create -f environment.yml
```

Activate the environment

```
conda activate conda-cuda
```

## Usage
### Local Training

Utilize the `run.py` script to initiate the training process locally. This script optimizes the model using the `Optuna` framework and saves the best model to the `saved_models` directory.

### Cloud Execution
The same project structure is expected when setting up an AWS Sagemaker project (or any cloud platform that supports PyTorch and CUDA). Hwever, no scripts for cloud execution have been provided in this repo. The following considerations need to be taken into account when setting up the cloud project:

- cloud bucket storage (e.g. S3, GCS, etc.)
- identifying compute resources and expressing them as code (e.g. ml.m4.xlarge)
- The estimator used to run the training job (e.g. `PyTorch` estimator in `AWS Sagemaker SDK`)
- the hyperparameter tuning framework being used right now is `optuna`. However, this can be replaced with in-built estimators in `AWS Sagemaker SDK`, `Vertex AI` or other frameworks like `ray` or `hyperopt`
- the inference is being deployed via `FAST API` using `uvicorn`, but in reality this can be replaced with any other framework 

## Network Architecture

The neural network consists of the following layers:

| Layer Name   | Type           | Description               | Dimension     |
|--------------|----------------|---------------------------|---------------|
| conv1        | Convolutional  | First layer               | 28x28x32      |
| conv2_1      | Convolutional  | Parallel layer            | 14x14x64      |
| conv2_2      | Convolutional  | Parallel layer            | 14x14x64      |
| conv3_1      | Convolutional  | Sequential layer          | 7x7x512       |
| conv3_2      | Convolutional  | Sequential layer          | 7x7x512       |
| fc1          | Fully Connected| -                         | 1000          |
| fc2          | Fully Connected| -                         | 500           |
| output_layer | Output         | -                         | 10

## Training results

Available on the local Optuna dashboard

run `optuna-dashboard sqlite:///db.sqlite3` to view training results

Current result status
```
Study statistics:
  Number of finished trials:  32
  Number of pruned trials:  4
  Number of complete trials:  21
Best trial:
  Value:  0.9928
  Params:
    batch_size: 33
    optimizer: Adam
    lr: 0.0003856386109765462
```

## Saved Model

The model file is saved in the `saved_models` directory. Due to size limitations, the actual model file has not been provided in the repo. The model file can be generated by running the `run.py` script (as long as the environment is set up correctly based on the package manager of choice). The model file is named `{DATETIME}-{HHMMSS}.pth`

In addition to the model file, the `saved_models` directory also contains a `model_info.json` file that contains the model hyperparameters.

## Deployment

The project currently employs `FAST API` coupled with Uvicorn for inference deployment which admittedly is a toy deployment.

## References
None yet.
