import torch as t
import torch
from torch import nn, optim
import torch.nn.functional as F

torch.manual_seed(40)

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(28, 28),
            nn.BatchNorm1d(28),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(28, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      
        out = self.mlp(x)
        #out = self.dropout(out)
        out = self.linear(out)
        out = self.sigmoid(out)
        return out
