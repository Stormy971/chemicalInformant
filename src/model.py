import torch
import torch.nn as nn

class ChemNet(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=1):
        super(ChemNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    def forward(self, x):
        return self.layers(x)
