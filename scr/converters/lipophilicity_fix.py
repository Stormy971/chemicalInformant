import pandas as pd

# Load Lipophilicity CSV
df = pd.read_csv("data/lipophilicity_raw.csv")

# Keep only the needed columns and rename them
df_clean = df[["CMPD_CHEMBLID", "exp", "RDKIT_SMILES"]].copy()
df_clean.rename(columns={
    "CMPD_CHEMBLID": "Compound ID",
    "exp": "measured log",
    "RDKIT_SMILES": "SMILES"
}, inplace=True)

# Save to new CSV
df_clean.to_csv("data/lipophilicity.csv", index=False)
print(f"Saved cleaned Lipophilicity CSV with {len(df_clean)} entries to data/lipophilicity_rdkit.csv")
