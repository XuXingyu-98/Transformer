import torch.nn as nn
import torch
import numpy as np
from torch import autograd

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model, requires_grad=True)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** (2 * i / d_model)))
                pe[pos, i + 1] = np.cos(pos / (10000 ** (2 * (i + 1) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * np.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + autograd.variable(self.pe[:, :seq_len]).cuda()
        return x