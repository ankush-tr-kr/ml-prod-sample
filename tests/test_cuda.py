import torch


def is_cuda_available():
    return torch.cuda.is_available()


def test_is_cuda_available():
    assert is_cuda_available() is False
