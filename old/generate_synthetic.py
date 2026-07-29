import pandas as pd
import numpy as np

# ==========================
# LOAD DATASET
# ==========================
df = pd.read_excel("crop-dataset.xlsx")

# ==========================
# SETTINGS
# ==========================
# TARGET_ROWS = 29400
TARGET_ROWS = 98000

# ==========================
# SAFE TRIANGULAR FUNCTION
# ==========================
def safe_triangular(low, high):
    
    """
    Generate value using triangular distribution.
    If low == high, return the value directly.
    """

    if pd.isna(low) or pd.isna(high):
        return np.nan

    low = float(low)
    high = float(high)

    if low >= high:
        return low

    mode = (low + high) / 2

    return np.random.triangular(low, mode, high)

# ==========================
# GENERATE DATA
# ==========================
synthetic_rows = []

samples_per_crop = max(1, TARGET_ROWS // len(df))

for _, row in df.iterrows():

    for _ in range(samples_per_crop):

        synthetic_rows.append({

            # Categorical Columns
            "CROPS": row["CROPS"],
            "TYPE_OF_CROP": row["TYPE_OF_CROP"],
            "SOIL": row["SOIL"],
            "SEASON": row["SEASON"],
            "SOWN": row["SOWN"],
            "HARVESTED": row["HARVESTED"],
            "WATER_SOURCE": row["WATER_SOURCE"],
            "Drought_Risk": row["Drought_Risk"],

            # Generated Numerical Columns
            "SOIL_PH": round(
                safe_triangular(
                    row["SOIL_PH_LOW"],
                    row["SOIL_PH_HIGH"]
                ), 2
            ),

            "CROP_DURATION": int(
                safe_triangular(
                    row["CROPDURATION_MIN"],
                    row["CROPDURATION_MAX"]
                )
            ),

            "TEMPERATURE": int(
                safe_triangular(
                    row["MIN_TEMP"],
                    row["MAX_TEMP"]
                )
            ),

            "WATER_REQUIRED": int(
                safe_triangular(
                    row["WATERREQUIRED_MIN"],
                    row["WATERREQUIRED_MAX"]
                )
            ),

            "RELATIVE_HUMIDITY": int(
                safe_triangular(
                    row["RELATIVE_HUMIDITY_MIN"],
                    row["RELATIVE_HUMIDITY_MAX"]
                )
            ),

            "N": int(
                safe_triangular(
                    row["N_MIN"],
                    row["N_MAX"]
                )
            ),

            "P": int(
                safe_triangular(
                    row["P_MIN"],
                    row["P_MAX"]
                )
            ),

            "K": int(
                safe_triangular(
                    row["K_MIN"],
                    row["K_MAX"]
                )
            )
        })

# ==========================
# CREATE DATAFRAME
# ==========================
synthetic_df = pd.DataFrame(synthetic_rows)

# ==========================
# EXACTLY 1000 ROWS
# ==========================
if len(synthetic_df) > TARGET_ROWS:
    synthetic_df = synthetic_df.sample(
        TARGET_ROWS,
        random_state=42
    )

while len(synthetic_df) < TARGET_ROWS:

    row = df.sample(1).iloc[0]

    synthetic_df.loc[len(synthetic_df)] = {

        "CROPS": row["CROPS"],
        "TYPE_OF_CROP": row["TYPE_OF_CROP"],
        "SOIL": row["SOIL"],
        "SEASON": row["SEASON"],
        "SOWN": row["SOWN"],
        "HARVESTED": row["HARVESTED"],
        "WATER_SOURCE": row["WATER_SOURCE"],
        "Drought_Risk": row["Drought_Risk"],

        "SOIL_PH": round(
            safe_triangular(
                row["SOIL_PH_LOW"],
                row["SOIL_PH_HIGH"]
            ), 2
        ),

        "CROP_DURATION": int(
            safe_triangular(
                row["CROPDURATION_MIN"],
                row["CROPDURATION_MAX"]
            )
        ),

        "TEMPERATURE": int(
            safe_triangular(
                row["MIN_TEMP"],
                row["MAX_TEMP"]
            )
        ),

        "WATER_REQUIRED": int(
            safe_triangular(
                row["WATERREQUIRED_MIN"],
                row["WATERREQUIRED_MAX"]
            )
        ),

        "RELATIVE_HUMIDITY": int(
            safe_triangular(
                row["RELATIVE_HUMIDITY_MIN"],
                row["RELATIVE_HUMIDITY_MAX"]
            )
        ),

        "N": int(
            safe_triangular(
                row["N_MIN"],
                row["N_MAX"]
            )
        ),

        "P": int(
            safe_triangular(
                row["P_MIN"],
                row["P_MAX"]
            )
        ),

        "K": int(
            safe_triangular(
                row["K_MIN"],
                row["K_MAX"]
            )
        )
    }

# ==========================
# SAVE FILES
# ==========================
synthetic_df.to_csv(
    "synthetic_crop_datasets1_2000.csv",
    index=False
)

synthetic_df.to_excel(
    "synthetic_crop_datasets1_2000.xlsx",
    index=False
)

print("✅ Synthetic Dataset Generated Successfully!")
print("Rows:", len(synthetic_df))
print("\nFirst 5 rows:")
print(synthetic_df.head())
print("\nFiles created:")
print("synthetic_crop_datasets1_100000.csv")
print("synthetic_crop_datasets1_100000.xlsx")