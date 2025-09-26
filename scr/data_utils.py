import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import numpy as np
from rdkit.Chem import rdFingerprintGenerator

# Use MorganGenerator instead of deprecated AllChem
def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fp = gen.GetFingerprint(mol)  # returns ExplicitBitVect
    return np.array(fp)


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
