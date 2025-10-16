import pandas as pd

# Paths to your CSV files
paths = {
    "delaney": "data/solubilityData/delaney.csv",
    "freesolv": "data/solubilityData/freesolv.csv",
    "lipo": "data/solubilityData/lipophilicity.csv",
    "bigsoldb": "data/solubilityData/bigsoldbv2.csv",  # new dataset
    "aqsoldb": "data/solubilityData/aqsolDB_v1.csv"
}

# Function to read CSV, fix headers, and select columns
def load_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")  # ensure proper encoding
    df.columns = df.columns.str.strip()           # remove spaces from headers
    # select only the columns we want
    df = df[["Compound ID", "measured log(solubility:mol/L)", "SMILES"]].copy()
    return df

# Load all datasets
df_delaney = load_csv(paths["delaney"])
df_freesolv = load_csv(paths["freesolv"])
df_lipo = load_csv(paths["lipo"])
df_bigsoldb = load_csv(paths["bigsoldb"])  # load BigSolDBv2
df_aqsoldb = load_csv(paths["aqsoldb"])

# Merge into a single DataFrame
df_merged = pd.concat([df_delaney, df_freesolv, df_lipo, df_bigsoldb, df_aqsoldb], ignore_index=True)

# Optional: remove duplicates if any (based on SMILES)
df_merged.drop_duplicates(subset="SMILES", inplace=True)

# Save the merged dataset
df_merged.to_csv("data/solubilityData/merged_solubility.csv", index=False)

print("Merged CSV saved as 'merged_solubility.csv'. Total compounds:", len(df_merged))
