import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self, in_channels, fs=32, kernel_size=25, pool_size=16, weight=0.001):
        super(CNN, self).__init__()
        self.receptive_field = kernel_size
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(in_channels, fs, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(fs)
        self.pool = nn.MaxPool1d(pool_size, stride=2, padding=pool_size // 2)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels, fs=32, ks_1=25, ps_1=16, ks_2=25, ps_2=16, weight=0.001):
        super(ResNet, self).__init__()
        self.cnn1 = CNN(in_channels, fs, ks_1, ps_1, weight)
        self.cnn2 = CNN(fs, fs, ks_2, ps_2, weight)
        self.shortcut_conv1 = nn.Conv1d(in_channels, fs, kernel_size=1, stride=2, padding=0)
        self.shortcut_conv2 = nn.Conv1d(fs, fs, kernel_size=1, stride=2, padding=0)
        self._initialize_weights()


    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.shortcut_conv1.weight, mode='fan_out', nonlinearity='relu')
        if self.shortcut_conv1.bias is not None:
            nn.init.constant_(self.shortcut_conv1.bias, 0)
        nn.init.kaiming_normal_(self.shortcut_conv2.weight, mode='fan_out', nonlinearity='relu')
        if self.shortcut_conv2.bias is not None:
            nn.init.constant_(self.shortcut_conv2.bias, 0)

    def forward(self, x):
        residual = self.shortcut_conv1(x)
        residual = self.shortcut_conv2(residual)
        out = self.cnn1(x)
        out = self.cnn2(out)
        if out.shape[-1] > residual.shape[-1]:
            out = out[:,:,:-1]
        elif out.shape[-1] < residual.shape[-1]:
            residual = residual[:,:,:-1]
        out += residual
        return out