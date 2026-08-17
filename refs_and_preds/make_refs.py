import pandas as pd

df = pd.read_csv("../data/processed/test.csv")
df = df[~df["tgt"].astype(str).str.contains("<<<<<<<|=======|>>>>>>>", na=False)]

reference_df = pd.DataFrame({
    "ID": range(1, len(df) + 1),
    "kashmiri_text": df["tgt"]
})
reference_df.to_csv("reference.csv", index=False)
print(f"reference_first.csv rows: {len(reference_df)}")
