import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from data_utils import smiles_to_fingerprint

# --- ChemNet definition matching saved model ---
class ChemNet(nn.Module):
    def __init__(self, input_dim=2048, hidden1_dim=1024, hidden2_dim=512, output_dim=1):
        super(ChemNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

# --- Load trained model ---
model_path = "models/solubility_model.pth"
model = ChemNet(input_dim=2048, hidden1_dim=1024, hidden2_dim=512, output_dim=1)
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)  # keys now match
model.eval()

print("ChemicalInformant Predictor")
print("Type a SMILES string to get predicted solubility (type 'exit' to quit).")

while True:
    smi = input("Enter SMILES: ").strip()
    if smi.lower() == "exit":
        break

    fp = smiles_to_fingerprint(smi)
    if fp is None:
        print("Invalid SMILES, try again.")
        continue

    fp_tensor = torch.tensor([fp], dtype=torch.float32)
    with torch.no_grad():
        pred = model(fp_tensor).item()
    print(f"Predicted solubility: {pred:.3f}")
