import os
from pathlib import Path

import torch
from model.network import MNISTClassifier
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTClassifier().to(device)
saved_models_dir = Path("saved_models")
model_files = list(saved_models_dir.glob("*.pth"))
latest_model_file = max(model_files, key=os.path.getctime)

print(f"Loading weights from {latest_model_file}...")
model.load_state_dict(torch.load(latest_model_file, map_location=device))
print("Weights loaded successfully.")
model.eval()  # Set the model to evaluation mode


def preprocess_image(image_path):
    """Preprocess an image file to the appropriate format for inference.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = Image.open(image_path).convert("L")  # Convert image to grayscale
    transform = transforms.Compose(
        [
            transforms.Resize(
                (28, 28)
            ),  # Resize image to the size expected by the model
            transforms.ToTensor(),  # Convert image to tensor
        ]
    )
    image = transform(image)
    image = image.unsqueeze(0)

    # DEBUG ONLY: display the image to be predicted
    # plt.imshow(image.squeeze().numpy(), cmap="gray")
    # plt.show()

    return image


def infer(image_path):
    """Perform inference on a single image and return the predicted class.

    Args:
        image_path (str): Path to the image file.

    Returns:
        int: Predicted class label.
    """
    image = preprocess_image(image_path)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
    print("Network output:", output)
    predicted_class = torch.argmax(
        output, dim=1
    ).item()  # Get the class with the highest score
    return predicted_class


# DEBUG ONLY: run inference on a single image from the command line
# Usage: python inference.py <path_to_image>
# uncomment the following lines to enable
# @click.command()
# @click.argument("image_path")
# def cli(image_path):
#     """Perform inference on a single image and print the predicted class."""
#     predicted_class = infer(image_path)
#     print(predicted_class)


# if __name__ == "__main__":
#     cli()
