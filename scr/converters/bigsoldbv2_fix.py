import pandas as pd
import os

# --- File paths ---
input_csv = "data/raw_data/BigSolDBv2.0.csv"  # adjust if needed
output_csv = "data/solubilityData/bigsoldbv2.csv"

# --- Load BigSolDB ---
df = pd.read_csv(input_csv)

# --- Prepare output DataFrame ---
# Use PubChem_CID if available for Compound ID, otherwise fallback to Compound_Name
def get_compound_id(row):
    if pd.notnull(row['PubChem_CID']):
        return str(int(row['PubChem_CID']))  # make integer string
    elif pd.notnull(row['Compound_Name']):
        return row['Compound_Name']
    else:
        return "Unknown"

converted_df = pd.DataFrame({
    "Compound ID": df.apply(get_compound_id, axis=1),
    "measured log(solubility:mol/L)": df["LogS(mol/L)"],
    "SMILES": df["SMILES_Solute"]
})

# --- Drop rows with missing SMILES or log solubility ---
converted_df = converted_df.dropna(subset=["SMILES", "measured log(solubility:mol/L)"])

# --- Save the converted CSV ---
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
converted_df.to_csv(output_csv, index=False)

print(f"✅ Conversion complete! Saved to {output_csv}")
print(f"Total molecules: {len(converted_df)}")
