import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        """
        Args:
            input_channels (int): Number of input channels (e.g., 1 for grayscale).
            hidden_channels (list): List of hidden channel sizes for convolutional layers.
        """
        super(CNN, self).__init__()
        layers = []
        in_channels = input_channels

        for out_channels in hidden_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)


        self.upsample = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], 1, kernel_size=3, stride=1, padding=1),
            nn.Upsample(size=(290, 290), mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        return x
