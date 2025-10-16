import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pandas as pd
import numpy as np
from model import ChemNet
from data_utils import smiles_to_fingerprint
import matplotlib.pyplot as plt

# --- Paths ---
test_csv = os.path.join("data", "solubilityData", "testing", "test_molecules.csv")
results_csv = os.path.join("data", "solubilityData", "testing", "test_results.csv")

# --- Load trained model ---
model = ChemNet(input_dim=2048, hidden1_dim=1024, hidden2_dim=512, output_dim=1)
model.load_state_dict(torch.load("models/solubility_model.pth"))
model.eval()

# --- Load test set ---
df_test = pd.read_csv(test_csv)

# --- Prepare fingerprints ---
X_test = []
valid_ids = []
valid_smiles = []
y_true = []
types = []
invalid_smiles = []

for idx, row in df_test.iterrows():
    smi = row["SMILES"]
    mol_type = row["Type"] if "Type" in row else "Unknown"
    try:
        fp = smiles_to_fingerprint(smi)
        X_test.append(fp)
        valid_ids.append(row["Compound ID"])
        valid_smiles.append(smi)
        types.append(mol_type)
        y_true.append(row["measured log(solubility:mol/L)"] if "measured log(solubility:mol/L)" in row else np.nan)
    except Exception as e:
        invalid_smiles.append(smi)
        print(f"Skipping invalid SMILES: {smi} ({e})")

X_test = torch.tensor(np.array(X_test), dtype=torch.float32)

# --- Predict ---
with torch.no_grad():
    preds = model(X_test).numpy().flatten()

# --- Build results DataFrame ---
df_results = pd.DataFrame({
    "Compound ID": valid_ids,
    "SMILES": valid_smiles,
    "Type": types,
    "Predicted log(solubility:mol/L)": preds,
    "Measured log(solubility:mol/L)": y_true
})

if invalid_smiles:
    print(f"\nSkipped {len(invalid_smiles)} invalid SMILES: {invalid_smiles}")

# --- Compute errors ---
df_results["Error"] = df_results["Predicted log(solubility:mol/L)"] - df_results["Measured log(solubility:mol/L)"]
df_results["AbsError"] = df_results["Error"].abs()

# --- Summary by type ---
type_summary = df_results.groupby("Type")["AbsError"].agg(["mean", "count"]).reset_index()
type_summary.rename(columns={"mean": "Mean AbsError", "count": "Num Molecules"}, inplace=True)
print("\n=== Error Summary by Molecule Type ===")
print(type_summary)

# --- Plot predicted vs actual solubility ---
plt.figure(figsize=(12,6))
colors = plt.cm.tab20.colors  # color palette for types
type_to_color = {t: colors[i % len(colors)] for i, t in enumerate(df_results["Type"].unique())}

for i, row in df_results.iterrows():
    plt.bar(i-0.2, row["Measured log(solubility:mol/L)"], width=0.4, color=type_to_color[row["Type"]], alpha=0.6)
    plt.bar(i+0.2, row["Predicted log(solubility:mol/L)"], width=0.4, color=type_to_color[row["Type"]], alpha=0.9)

plt.ylabel("log(solubility:mol/L)")
plt.xlabel("Molecules")
plt.title("Predicted vs Actual Solubility (Color-coded by Type)")
plt.xticks([])
plt.legend(handles=[plt.Rectangle((0,0),1,1,color=c, alpha=0.7) for t,c in type_to_color.items()],
           labels=type_to_color.keys(), title="Molecule Type", bbox_to_anchor=(1.05,1))
plt.tight_layout()
plt.show()

# --- Save full results ---
df_results.to_csv(results_csv, index=False)
print(f"\n✅ Full test results saved to '{results_csv}'")
