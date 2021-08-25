import torch
import torch.nn as nn
from Encoder_and_Decoder import Encoder, Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_outputs = self.decoder(trg, e_outputs, src_mask, trg_mask)
        outputs = self.out(d_outputs)
        return outputs