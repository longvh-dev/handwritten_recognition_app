import torch
from torch import nn

"""
This is a Convolutional Neural Network (CNN) class that inherits from PyTorch's nn.Module class.

The CNN class defines a basic CNN model with two convolutional layers, a max pooling layer, and two fully connected layers.

Attributes:
conv1 (nn.Conv2d): The first convolutional layer which takes a single-channel input and applies 32 filters.
pool (nn.MaxPool2d): The max pooling layer which reduces the spatial dimensions of the input.
conv2 (nn.Conv2d): The second convolutional layer which takes the 32-channel output from the previous layer as input.
fc1 (nn.Linear): The first fully connected layer which takes the flattened output from the previous layer as input and outputs 128 features.
fc2 (nn.Linear): The second fully connected layer which takes the 128 features as input and outputs a score for each class in num_classes.
"""


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Defines the forward pass of the CNN model.

        Parameters:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output tensor after passing through the model.
        """
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
