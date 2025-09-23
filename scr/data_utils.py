import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import numpy as np

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    """Convert SMILES to Morgan fingerprint (bit vector)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def load_dataset(csv_path, smiles_col="smiles", target_col="measured log solubility in mols per litre"):
    """Load CSV and return features (X) and targets (y)."""
    df = pd.read_csv(csv_path)
    X = []
    y = []

    for _, row in df.iterrows():
        fp = smiles_to_fingerprint(row[smiles_col])
        if fp is not None:
            X.append(fp)
            y.append(row[target_col])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y
