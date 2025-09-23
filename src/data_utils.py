from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np

def smiles_to_fingerprint(smiles: str, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

def load_dataset(csv_path: str, smiles_col="smiles", target_col="property"):
    df = pd.read_csv(csv_path)
    X = []
    y = []
    for _, row in df.iterrows():
        fp = smiles_to_fingerprint(row[smiles_col])
        if fp is not None:
            X.append(fp)
            y.append(row[target_col])
    return np.array(X), np.array(y)
