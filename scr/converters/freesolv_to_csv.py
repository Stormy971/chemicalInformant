import pickle
import pandas as pd

with open("data/freeSolv_data_Solubility.pickle", "rb") as f:
    data = pickle.load(f, encoding='latin1')

rows = []
for entry_id, entry in data.items():
    smiles = entry["smiles"]
    expt = entry["expt"]  # hydration free energy
    rows.append({
        "Compound ID": entry_id,
        "measured log(solubility:mol/L)": expt,
        "SMILES": smiles
    })

df = pd.DataFrame(rows)
df.to_csv("data/freesolv.csv", index=False)
print(f"Saved {len(df)} entries to data/freesolv.csv")
