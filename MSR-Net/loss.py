import torch
from torch import nn

class FrobiniusLoss(nn.Module):
    def __init__(self):
        super(FrobiniusLoss, self).__init__()

    def forward(self, output, label):
        return torch.sum((output - label)**2) / len(output)