import pandas as pd

# Paths to your datasets
paths = {
    "delaney": "data/solubilityData/delaney.csv",
    "freesolv": "data/solubilityData/freesolv.csv",
    "lipophilicity": "data/solubilityData/lipophilicity.csv"
}

# Load and standardize Delaney dataset
df_delaney = pd.read_csv(paths["delaney"])
df_delaney = df_delaney[["Compound ID", "measured log(solubility:mol/L)", "SMILES"]].copy()

# Load and standardize FreeSolv dataset
df_freesolv = pd.read_csv(paths["freesolv"])
df_freesolv = df_freesolv[["Compound ID", "measured log(solubility:mol/L)", "SMILES"]].copy()

# Load and standardize Lipophilicity dataset
df_lipo = pd.read_csv(paths["lipophilicity"])
df_lipo = df_lipo[["Compound ID", "measured log(solubility:mol/L)", "SMILES"]].copy()

# Concatenate all datasets
df_merged = pd.concat([df_delaney, df_freesolv, df_lipo], ignore_index=True)

# Optional: drop duplicate compounds based on SMILES
df_merged.drop_duplicates(subset="SMILES", inplace=True)

# Save merged CSV
output_path = "data/merged_solubility.csv"
df_merged.to_csv(output_path, index=False)

print(f"Merged dataset saved to {output_path}")
print(f"Total compounds: {len(df_merged)}")
