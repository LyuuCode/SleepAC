import torch
import torch.nn as nn

from layers.resnet_encoder import ResNetEncoder


class TemporalEncoder(nn.Module):
    def __init__(self, in_channels=9, out_channels=256):
        super(TemporalEncoder, self).__init__()
        self.in_channels = in_channels
        self.encoder = ResNetEncoder(in_channels, out_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.encoder(x)

        x = self.global_avg_pool(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x