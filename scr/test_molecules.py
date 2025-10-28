# scr/test_molecules.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ------------------------------
# 1. Load data
# ------------------------------
df = pd.read_csv("data/solubilityData/merged_solubility_full.csv")

# Keep only valid SMILES
df = df[df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x) is not None)].reset_index(drop=True)

# Target column
target_col = 'measured log(solubility:mol/L)'

# ------------------------------
# 2. Load model and scaler
# ------------------------------
class SolubilityNet(torch.nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1024):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = torch.nn.Linear(hidden_dim, 1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.dropout(self.relu(self.layer2(out)))
        out = self.layer3(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SolubilityNet().to(device)
model.load_state_dict(torch.load("models/solubility_model.pth", map_location=device))
model.eval()

scaler: StandardScaler = torch.load("models/scaler.pth")

# ------------------------------
# 3. Prepare features
# ------------------------------
def mol_features(smiles, nBits=2048, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

X = np.array([mol_features(s) for s in df['SMILES']])
X = scaler.transform(X)
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_true = df[target_col].values

# ------------------------------
# 4. Predict
# ------------------------------
with torch.no_grad():
    y_pred = model(X_tensor).cpu().numpy().flatten()

# ------------------------------
# 5. Metrics
# ------------------------------
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ------------------------------
# 6. Scatter plot of predicted vs measured
# ------------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.6)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel("Measured")
plt.ylabel("Predicted")
plt.title("Predicted vs Measured Solubility")
plt.grid(True)
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/predicted_vs_measured.png")
plt.show()
