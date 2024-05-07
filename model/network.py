"""
This file contains the network architecture for the MNIST classifier.
"""

from torch import nn


# pylint: disable=too-many-instance-attributes
class MNISTClassifier(nn.Module):
    """CNN model for MNIST classification."""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
        )

        # Parallel layers; using groups parameter to split the input into groups
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1, groups=2),
            nn.ReLU(),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1, groups=2),
            nn.ReLU(),
        )

        # Sequential layers
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
        )

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=1000),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU(),
        )

        # Output layer
        self.output_layer = nn.Linear(in_features=500, out_features=10)

    def forward(self, x):  # pylint: disable=invalid-name
        """Forward pass of the network."""
        x = self.conv1(x)
        x = nn.MaxPool2d(kernel_size=(2, 2), stride=2)(x)  # downsampling to 14x14

        # Parallel layers
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x = nn.MaxPool2d(kernel_size=(2, 2), stride=2)(
            x2_1 + x2_2
        )  # Combining and downsampling to 7x7

        # Sequential layers
        x = self.conv3_1(x)
        x = self.conv3_2(x)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)

        # Output layer
        x = self.output_layer(x)

        return x


net = MNISTClassifier()
