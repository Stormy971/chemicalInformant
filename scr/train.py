# scr/train.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Prevent OpenMP conflicts

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------
# 1. Load and clean data
# ------------------------------
df = pd.read_csv("data/solubilityData/merged_solubility_full.csv")

# Drop rows with invalid SMILES
def is_valid_smiles(smi):
    return Chem.MolFromSmiles(smi) is not None

df['valid'] = df['SMILES'].apply(is_valid_smiles)
df = df[df['valid']].reset_index(drop=True)

# Features: Morgan Fingerprints
def mol_features(smiles, nBits=2048, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((1,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# Target column
target_col = 'measured log(solubility:mol/L)'

# ------------------------------
# 2. Dataset
# ------------------------------
class MoleculeDataset(Dataset):
    def __init__(self, df):
        self.X = np.array([mol_features(smi) for smi in df['SMILES']], dtype=np.float32)
        self.y = df[target_col].values.astype(np.float32).reshape(-1, 1)
        
        # Normalize features
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# ------------------------------
# 3. Model
# ------------------------------
class SolubilityNet(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1024):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Optional regularization
        
    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.dropout(self.relu(self.layer2(out)))
        out = self.layer3(out)
        return out

# ------------------------------
# 4. Prepare DataLoaders
# ------------------------------
dataset = MoleculeDataset(df)
train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)

train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=64, shuffle=True)
val_loader   = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=64, shuffle=False)

# ------------------------------
# 5. Training
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SolubilityNet(input_dim=2048).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses, val_losses = [], []

for epoch in range(1, 51):
    # Training
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch:02d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

# ------------------------------
# 6. Save model and scaler
# ------------------------------
torch.save(model.state_dict(), "models/solubility_model.pth")
torch.save(dataset.scaler, "models/scaler.pth")

# ------------------------------
# 7. Plot loss curves
# ------------------------------
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/loss_curve.png")
plt.show()
