import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        if in_features != out_features:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        res = x if self.residual is None else self.residual(x)
        return self.relu(out + res)

class ChemNet(nn.Module):
    def __init__(self, input_size=512):
        super().__init__()
        self.block1 = ResidualBlock(input_size, 256)
        self.block2 = ResidualBlock(256, 128)
        self.block3 = ResidualBlock(128, 64)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.fc_out(x)
