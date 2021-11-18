import torch
import torch.nn as nn
import torch.functional as F


class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r] mels
    output --- [N. ref_enc_gru_size]
    '''

    def __init__(self, model_config, preprocess_config):

        super().__init__()
        self.model_config = model_config
        self.preprocess_config = preprocess_config
        K = len(self.model_config['gst']['ref_enc_filters'])
        filters = [1] + self.model_config['gst']['ref_enc_filters']
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i+1],
                           kernel_size=(3,3),
                           stride=(2,2),
                           padding=(1,1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=self.model_config["gst"]["ref_enc_filters"][i]) for i in range(K)])

        out_channels = self.calculate_channels(self.preprocess_config['preprocessing']['mel']['n_mel_channels'], 3, 2, 1, K)
        self.gru = nn.GRU(input_size=self.model_config['gst']['ref_enc_filters'][-1] * out_channels,
                          hidden_size=self.model_config['gst']['E']//2,
                          batch_first=True
                )
    def forward(self, inputs):
        N =  inputs.size(0)
        out = inputs.view(N, 1, -1, self.preprocess_config['preprocessing']['mel']['n_mel_channels'])
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out= out.transpose(1, 2)
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)

        self.gru.flatten_parameters()
        memory, out = self.gru(out)

        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

