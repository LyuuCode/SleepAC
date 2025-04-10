import torch
import torch.nn as nn

from layers.residual_block import ResNet


# ResNetEncoder: Encapsulating multiple ResNet blocks
class ResNetEncoder(nn.Module):
    def __init__(self, channels=9, out_channels=128):
        super(ResNetEncoder, self).__init__()
        ch3 = out_channels // 2
        ch2 = out_channels // 4
        ch1 = out_channels // 8
        self.resnet1 = ResNet(channels, ch1)
        self.resnet2 = ResNet(ch1, ch2)
        self.resnet3 = ResNet(ch2, ch3)
        self.resnet4 = ResNet(ch3, out_channels)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.resnet1(x)
        x = self.dropout(x)
        x = self.resnet2(x)
        x = self.dropout(x)
        x = self.resnet3(x)
        x = self.dropout(x)
        x = self.resnet4(x)
        return x