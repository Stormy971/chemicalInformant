import torch
from model import ChemNet
from data_utils import smiles_to_fingerprint

# --- Load trained model ---
model = ChemNet(input_dim=2048, hidden1_dim=1024, hidden2_dim=512, output_dim=1)
model.load_state_dict(torch.load("models/chemnet_weights.pth"))  # make sure you saved your trained weights
model.eval()

print("ChemicalInformant Predictor")
print("Type a SMILES string to get predicted solubility (type 'exit' to quit).")

while True:
    smi = input("Enter SMILES: ").strip()
    if smi.lower() == "exit":
        break

    # Convert SMILES to fingerprint
    fp = smiles_to_fingerprint(smi)
    if fp is None:
        print("Invalid SMILES, try again.")
        continue

    # Convert to tensor
    fp_tensor = torch.tensor([fp], dtype=torch.float32)

    # Predict
    with torch.no_grad():
        pred = model(fp_tensor).item()

    print(f"Predicted solubility: {pred:.3f}")
