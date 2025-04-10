import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.temporal_encoder import TemporalEncoder
from layers.frequency_encoder import FrequencyEncoder


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        in_channels = configs.num_nodes
        out_channels = configs.out_channels
        seq_len = configs.seq_len
        freq_len = configs.freq_len
        self.temporalEncoder = TemporalEncoder(in_channels=in_channels, out_channels=out_channels)
        self.frequencyEncoder = FrequencyEncoder(in_channels=in_channels, out_channels=out_channels, freq_len=freq_len)
        self.deconv = nn.ConvTranspose1d(in_channels=out_channels, out_channels=in_channels, kernel_size=seq_len)
        self.batch_norm = nn.BatchNorm1d(out_channels*2)
        self.fc = nn.Linear(out_channels*2, configs.num_class)
        self.dropout = nn.Dropout(p=0.3)
    def forward(self, x, is_train=False, is_al=False):
        data = x['data']
        stft = x['stft']

        xt = self.temporalEncoder(data)
        xf = self.frequencyEncoder(stft)
        x_enc = torch.cat((xt, xf), dim=-1)
        x_enc = self.batch_norm(x_enc)
        x_enc = self.dropout(x_enc)
        x_out = self.fc(x_enc)

        if(is_train):
            data_mask = x['data_mask']
            xt_mask = self.temporalEncoder(data * data_mask)
            xt_mask = xt_mask.unsqueeze(2)
            x_recon = self.deconv(xt_mask).permute(0, 2, 1)
            return x_enc, x_out, x_recon
        elif(is_al):
            return x_out, x_enc
        return x_out