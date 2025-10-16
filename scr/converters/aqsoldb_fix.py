import pandas as pd
import numpy as np

# Input CSV
input_csv = "data/raw_data/AqSolDB_v1.0_min.csv"

# Output CSV
output_csv = "data/solubilityData/aqsolDB_v1.csv"

# Load the dataset
df = pd.read_csv(input_csv)

# Ensure 'Solubility' column exists
if "Solubility" not in df.columns:
    raise ValueError("Input CSV must have a 'Solubility' column")

# Convert solubility to log scale (mol/L)
# Assume Solubility is in mol/L already; if in mg/L or other units, adjust accordingly
df["measured log(solubility:mol/L)"] = np.log10(df["Solubility"].replace(0, np.nan))

# Fill Compound ID with Name if ID is not provided
if "ID" in df.columns:
    df["Compound ID"] = df["ID"]
else:
    df["Compound ID"] = df["Name"]

# Prepare output dataframe
df_out = df[["Compound ID", "measured log(solubility:mol/L)", "SMILES"]].copy()

# Drop rows with NaN values (e.g., zero or missing solubility)
df_out.dropna(subset=["measured log(solubility:mol/L)", "SMILES"], inplace=True)

# Save to CSV
df_out.to_csv(output_csv, index=False)
print(f"✅ Converted dataset saved to '{output_csv}'")
