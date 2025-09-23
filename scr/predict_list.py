import torch
from model import ChemNet
from data_utils import smiles_to_fingerprint

# --- Load trained model ---
model = ChemNet(input_dim=2048, hidden1_dim=1024, hidden2_dim=512, output_dim=1)
model.load_state_dict(torch.load("models/chemnet_weights.pth"))
model.eval()

print("ChemicalInformant Predictor")
print("Enter one or more SMILES separated by commas (type 'exit' to quit).")

while True:
    smi_input = input("Enter SMILES: ").strip()
    if smi_input.lower() == "exit":
        break

    # Split multiple SMILES
    smi_list = [s.strip() for s in smi_input.split(",") if s.strip()]

    # Convert each SMILES to fingerprint
    X_new = []
    valid_smiles = []
    for smi in smi_list:
        fp = smiles_to_fingerprint(smi)
        if fp is None:
            print(f"Invalid SMILES: {smi}, skipping.")
        else:
            X_new.append(fp)
            valid_smiles.append(smi)

    if not X_new:
        print("No valid SMILES to predict. Try again.")
        continue

    # Convert to tensor
    X_tensor = torch.tensor(X_new, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        preds = model(X_tensor).numpy().flatten()

    # Print results
    for smi, pred in zip(valid_smiles, preds):
        print(f"SMILES: {smi}, Predicted solubility: {pred:.3f}")
