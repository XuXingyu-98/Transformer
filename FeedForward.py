import torch.nn as nn
import torch

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2028, droput=0.1):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(droput)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.droput(torch.relu_(self.linear1(x)))
        x = self.linear2(x)
        return x