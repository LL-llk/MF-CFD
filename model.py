# model.py
import torch
import torch.nn as nn

class ResidualSliceCNN(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1, stride=2),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, 1, 3, padding=1)
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.bottleneck(z)
        return self.decoder(z)
