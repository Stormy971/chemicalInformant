import torch
import torch.nn as nn

class ChemNet(nn.Module):
    def __init__(self, input_dim=2048, hidden1_dim=1024, hidden2_dim=512, output_dim=1):
        """
        Neural network for predicting chemical properties from Morgan fingerprints.
        
        Args:
            input_dim (int): Size of input features (fingerprint length, default 2048)
            hidden1_dim (int): Number of neurons in hidden layer 1
            hidden2_dim (int): Number of neurons in hidden layer 2
            output_dim (int): Output size (1 for regression)
        """
        super(ChemNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),  # Hidden layer 1
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden1_dim, hidden2_dim), # Hidden layer 2
            nn.ReLU(),
            nn.Linear(hidden2_dim, output_dim)   # Output layer
        )

    def forward(self, x):
        return self.layers(x)
