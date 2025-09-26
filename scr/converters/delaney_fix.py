import pandas as pd

# Load Delaney CSV
df = pd.read_csv("data/delaney.csv")

# Drop the ESOL predicted column
df = df.drop(columns=["ESOL predicted log(solubility:mol/L)"])

# Save to a new CSV
df.to_csv("data/delaney_clean.csv", index=False)
print(f"Saved cleaned Delaney CSV with {len(df)} entries to data/delaney_clean.csv")
