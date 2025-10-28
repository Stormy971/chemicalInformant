import os
import pandas as pd

# === Input paths ===
data_dir = "data/solubilityData"

# Define dataset files
datasets = {
    "Delaney": os.path.join(data_dir, "delaney.csv"),
    "FreeSolv": os.path.join(data_dir, "freesolv.csv"),
    "Lipophilicity": os.path.join(data_dir, "lipophilicity.csv"),
    "AQSolDB": os.path.join(data_dir, "AQSolDBData.csv"),
    "BigSolDB": os.path.join(data_dir, "bigsoldbv2.csv"),
}

merged_data = []

# === Load datasets ===
for name, path in datasets.items():
    if os.path.exists(path):
        print(f"✅ Loaded {name} dataset from {path}")
        df = pd.read_csv(path)

        # Normalize column names
        df = df.rename(columns={
            "Compound ID": "Compound ID",
            "measured log(solubility:mol/L)": "measured log(solubility:mol/L)",
            "SMILES": "SMILES"
        })

        # Ensure required columns exist
        if not {"Compound ID", "measured log(solubility:mol/L)", "SMILES"}.issubset(df.columns):
            print(f"⚠️ {name} missing one or more required columns, skipping")
            continue

        df["Source"] = name
        merged_data.append(df)
    else:
        print(f"⚠️ Missing dataset: {name} ({path})")

# === Merge all ===
if not merged_data:
    raise FileNotFoundError("❌ No datasets found!")

merged_df = pd.concat(merged_data, ignore_index=True)

# Do NOT drop duplicates or missing values — preserve all
output_path = os.path.join(data_dir, "merged_solubility_full.csv")
os.makedirs(data_dir, exist_ok=True)
merged_df.to_csv(output_path, index=False)

print(f"\n✅ Merged dataset saved: {output_path}")
print(f"📊 Total entries: {len(merged_df)}")
print(f"📁 Sources merged: {', '.join(datasets.keys())}")
