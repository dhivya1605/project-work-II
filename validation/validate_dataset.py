from config import INPUT_FILE
from utils.excel_reader import load_dataset

df = load_dataset(INPUT_FILE)

range_columns = [
    ("SOIL_PH_LOW", "SOIL_PH_HIGH"),
    ("CROPDURATION_MIN", "CROPDURATION_MAX"),
    ("MIN_TEMP", "MAX_TEMP"),
    ("WATERREQUIRED_MIN", "WATERREQUIRED_MAX"),
    ("RELATIVE_HUMIDITY_MIN", "RELATIVE_HUMIDITY_MAX"),
    ("N_MIN", "N_MAX"),
    ("P_MIN", "P_MAX"),
    ("K_MIN", "K_MAX"),
]

print("\nChecking dataset...\n")

errors = 0

for _, row in df.iterrows():
    for low, high in range_columns:
        if row[low] > row[high]:
            print(f"{row['CROPS']} -> {low} > {high}")
            errors += 1

if errors == 0:
    print("All ranges are valid.")
else:
    print(f"\nFound {errors} invalid ranges.")