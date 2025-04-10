import torch
import torch.nn as nn

from layers.dilated_conv_encoder import DilatedConvEncoder


class FrequencyEncoder(nn.Module):
    def __init__(self, in_channels=9, out_channels=256, hidden_dims=128, depth=4, freq_len=144):
        super(FrequencyEncoder, self).__init__()
        self.encoder = DilatedConvEncoder(in_channels, out_channels, hidden_dims, depth)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(out_channels*freq_len, out_channels)
    def forward(self, x):
        enc = self.encoder(x)
        enc = self.flatten(enc)
        enc = self.fc(enc)
        return enc