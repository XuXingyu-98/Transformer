import torch
import torch.nn as nn
from Normalization import Norm
from MultiheadAttention import MultiHeadAttention
from FeedForward import FeedForward
import copy

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.attention = MultiHeadAttention(d_model, heads)
        self.feedForward = FeedForward(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.attention(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.feedForward(x2))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.norm3 = Norm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.attention1 = MultiHeadAttention(heads, d_model)
        self.attention2 = MultiHeadAttention(heads, d_model)
        self.feedForward = self.feedForward(d_model).cuda()

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm1(x)
        x = x + self.dropout_1(self.attention1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attention2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.feedForward(x2))
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])