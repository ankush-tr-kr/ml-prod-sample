import torch
from model.network import MNISTClassifier


def test_MNISTClassifier():
    # Create a random tensor with the shape [batch_size, channels, height, width]
    x = torch.randn((1, 1, 28, 28))

    # Instantiate your model
    model = MNISTClassifier()

    # Get the output of the model
    output = model(x)

    # Check that the output has the correct shape
    assert output.shape == torch.Size(
        [1, 10]
    ), f"Expected output shape [1, 10], but got {output.shape}"
