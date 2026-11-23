import torch
from torch import nn

class L2SP(nn.Module):
    def __init__(self, initial_weights, alpha=0.01):
        super(L2SP, self).__init__()
        self.initial_weights = [w.detach().clone() for w in initial_weights]
        self.alpha = alpha

    def forward(self, model):
        total_squared_distance = 0.0
        for param, initial_param in zip(model.parameters(), self.initial_weights):
            total_squared_distance += ((param - initial_param)**2).sum()
        return (self.alpha / 2) * total_squared_distance