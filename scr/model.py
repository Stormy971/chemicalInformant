import torch
import torch.nn as nn

class ChemNet(nn.Module):
    def __init__(self, input_dim=2048, hidden1_dim=1024, hidden2_dim=512, output_dim=1):
        super(ChemNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x).squeeze()
