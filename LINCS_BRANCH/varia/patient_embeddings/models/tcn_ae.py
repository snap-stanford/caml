"""TCN encoder inspired from here: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py"""
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from embeddings.models.base_model import BaseModel

# TODO: is this correct?
class Chomp1d(nn.Module):
    """utility module to ensure causality (also by locuslab)"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlockEncoder(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlockEncoder, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    def __init__(
            self, 
            num_inputs=94623, 
            num_channels=(256,128,64), 
            latent_dim=64, 
            kernel_size=5, 
            dropout=0.2
        ):
        super(TCNEncoder, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_channels[0] if i == 0 else num_channels[i-1]
            #in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlockEncoder(in_channels, out_channels, 
                kernel_size, stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size, dropout=dropout)
            ]

        self.first_linear = nn.Linear(num_inputs, num_channels[0])
        self.network = nn.Sequential(*layers)
        self.last_linear = nn.Linear(num_channels[-1],latent_dim)

    def forward(self, x):
        x = self.first_linear(x.permute(0, 2, 1)).permute(0,2,1) # applied to channel dimensions
        x = self.network(x)
        x = self.last_linear(x.permute(0, 2, 1)).permute(0,2,1) # applied to channel dimensions
        return x


class TemporalBlockDecoder(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlockDecoder, self).__init__()
        self.conv1 = weight_norm(nn.ConvTranspose1d(n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.ConvTranspose1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        #self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
        #                         self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNDecoder(nn.Module):
    def __init__(self, num_inputs=64, num_channels=(64,128, 256), output_dim=94623, kernel_size=5, dropout=0.2):
        super(TCNDecoder, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlockDecoder(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1],output_dim)

    def forward(self, x):
        x = self.network(x)
        return self.linear(x.permute(0, 2, 1)).permute(0,2,1) # apply final linear to channel dimensions


class TCNAutoencoder(BaseModel):
    """ TCN Autoencoder Model """
    def __init__(self, encoder_params: dict={}, decoder_params: dict={}, **kwargs):
        super().__init__(**kwargs)
        self.encoder = TCNEncoder(**encoder_params)
        self.decoder = TCNDecoder(**decoder_params)
        self.save_hyperparameters()

    def forward(self, x, lengths): # lenghts not used here
        x = self.encoder(x)
        return self.decoder(x)

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        #TODO: add model specific parser 
        return parser

