import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

# Make Morgan fingerprints for molecules
def mol_features(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[Warning] Invalid SMILES skipped: {smiles}")
        return None  # Return None instead of raising
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# Standard scaler
class StandardScalerTorch:
    def __init__(self, X):
        self.mean = torch.tensor(X.mean(axis=0), dtype=torch.float32)
        self.std = torch.tensor(X.std(axis=0), dtype=torch.float32)

    def transform(self, X):
        return (X - self.mean.numpy()) / self.std.numpy()

    def inverse_transform(self, X_scaled):
        return X_scaled * self.std.numpy() + self.mean.numpy()
