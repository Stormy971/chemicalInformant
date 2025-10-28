import os
import pandas as pd

# Input and output directories
input_dir = "data/raw_data/AQSOLDatasets"
output_dir = "data/solubilityData"
output_file = os.path.join(output_dir, "AQSolDBData.csv")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Expected columns in the original files
columns_expected = ["ID", "Name", "InChI", "InChIKey", "SMILES", "Solubility", "Prediction"]

# Initialize list to collect all data
all_data = []

# Loop through AQSolDB_A to AQSolDB_I
for letter in "ABCDEFGHI":
    file_name = f"dataset-{letter}.csv"
    file_path = os.path.join(input_dir, file_name)

    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        continue

    print(f"Processing {file_name}...")

    # Read CSV
    df = pd.read_csv(file_path)

    # Verify headers (optional safety check)
    if not set(["ID", "SMILES", "Solubility"]).issubset(df.columns):
        print(f"❌ Unexpected format in {file_name}, skipping.")
        continue

    # Rename columns to match your target format
    df_converted = pd.DataFrame({
        "Compound ID": df["ID"],
        "measured log(solubility:mol/L)": df["Solubility"],
        "SMILES": df["SMILES"]
    })

    all_data.append(df_converted)

# Merge all into one DataFrame
if all_data:
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"✅ Saved merged file: {output_file} ({len(merged_df)} entries)")
else:
    print("⚠️ No data merged — check your input files.")
