import torch
from config import *
from torch import nn
import torch.nn.functional as F

class MultiScaleLogTransform(nn.Module):
    def __init__(self, n, v):
        super(MultiScaleLogTransform, self).__init__()
        self.msl_conv = nn.Conv2d(3 * n, 3, (3, 3), padding = 1)
        self.v = v

    def forward(self, x):
        M_i = [torch.log(1 + i * x) / torch.log(1 + torch.tensor(i)) for i in self.v]
        M = torch.concat(M_i, axis = 1)
        return F.relu(self.msl_conv(M))

class DepthofConvolution(nn.Module):
    def __init__(self, k):
        super(DepthofConvolution, self).__init__()
        self.H = nn.ModuleList([nn.Conv2d(3, 3, (3, 3), padding = 1) for _ in range(k)])
        self.dim_red = nn.Conv2d(3 * k, 3, (1, 1))

    def forward(self, x):
        X = x.clone()
        H_i = []
        for conv in self.H:
            x = F.relu(conv(x))
            H_i.append(x)
        
        H = torch.concat(H_i, axis = 1)
        H = self.dim_red(H)
        return X - H

class ColorRestoration(nn.Module):
    def __init__(self):
        super(ColorRestoration, self).__init__()
        self.color_rest = nn.Conv2d(3, 3, (1, 1))

    def forward(self, x):
        return self.color_rest(x)

class MSRNet(nn.Module):
    def __init__(self, n, k, v):
        super(MSRNet, self).__init__()
        self.msrnet = nn.Sequential(MultiScaleLogTransform(n,v),
                                    DepthofConvolution(k),
                                    ColorRestoration())

    def forward(self, x):
        return self.msrnet(x)