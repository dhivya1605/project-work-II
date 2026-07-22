import pandas as pd

df = pd.read_csv("output/lhs_dataset.csv")

print("Total Rows :", len(df))
print("Total Columns :", len(df.columns))

print("\nRows per Crop:\n")

print(df["CROPS"].value_counts())

print("\nMissing Values:\n")

print(df.isnull().sum())

print("\nDuplicate Rows :", df.duplicated().sum())