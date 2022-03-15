import numpy as np
import torch
from torch import nn

class BiLSTM(nn.Module):
    def __init__(
        self, 
        embedding_dim, 
        hidden_dim, 
        num_layers,
        dropout,
        tagset_size, 
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.tagset_size = tagset_size

        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, embeds):
        bilstm_outputs, _  = self.bilstm(embeds)
        tags = self.hidden2tag(bilstm_outputs)
        return tags

if __name__=="__main__":
    model = BiLSTM(embedding_dim=300, hidden_dim=1024, num_layers=1, dropout=0.0, tagset_size=14)    
    model.eval()
    with torch.no_grad():
        input_sample = torch.randn(32, 50, 300)
        output_sample = model(input_sample)