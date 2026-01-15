import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Small CNN for 28x28 grayscale MNIST images.
    Outputs logits over 10 classes (digits 0–9).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # input: (1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # -> (32, 28, 28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> (64, 14, 14) after pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)  # (batch, 32, 14, 14)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)  # (batch, 64, 7, 7)

        x = self.dropout(x)
        x = torch.flatten(x, 1)  # (batch, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)          # logits
        return x
