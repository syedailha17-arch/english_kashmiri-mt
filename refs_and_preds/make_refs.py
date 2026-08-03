import pandas as pd

df = pd.read_csv("../data/processed/train.csv")
df["tgt"].to_csv("refs.txt", index=False, header=False)
