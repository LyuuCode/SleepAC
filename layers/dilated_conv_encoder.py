import torch
from torch import nn

from layers.dilated_conv_block import DilatedConvBlock


class DilatedConvEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, kernel_size=7):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvBlock(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=kernel_size
        )

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):  # x: B x T x input_dims
        x = self.input_fc(x)  # B x T x Ch

        x = x.transpose(1, 2)  # B x Ch x T
        x = self.dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        return x

