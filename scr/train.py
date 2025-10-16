import sys, os
sys.path.append(os.path.dirname(__file__))  # allows importing local modules
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model import ChemNet  # <-- Use your ChemNet model

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class ChemDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.smiles = df["SMILES"].values
        self.labels = df["measured log(solubility:mol/L)"].values.astype(np.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        X = self.mol_to_fp(self.smiles[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return X, y

    def mol_to_fp(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return torch.zeros(2048)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        arr = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return torch.tensor(arr, dtype=torch.float32)


def train_model(csv_file, epochs=100, batch_size=32, lr=1e-3, device="cpu"):
    dataset = ChemDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize ChemNet
    model = ChemNet(input_dim=2048, hidden1_dim=1024, hidden2_dim=512, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    all_epoch_losses = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X.size(0)

        epoch_loss /= len(dataset)
        all_epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1} completed: Loss = {epoch_loss:.4f}")

    # Print all epoch losses at the very end
    print("\n===== Epoch Losses =====")
    for i, loss in enumerate(all_epoch_losses, 1):
        print(f"Epoch {i:3d}: Loss = {loss:.4f}")

    # Evaluate on the full dataset
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            outputs = model(X)
            y_true.extend(y.numpy())
            y_pred.extend(outputs.cpu().numpy())

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n===== Final Test Set Evaluation =====")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # Save trained model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/solubility_model.pth")
    print("✅ Model saved as models/solubility_model.pth")


if __name__ == "__main__":
    csv_path = "data/solubilityData/merged_solubility.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    train_model(csv_path, epochs=25, batch_size=32, lr=1e-3, device=device)
