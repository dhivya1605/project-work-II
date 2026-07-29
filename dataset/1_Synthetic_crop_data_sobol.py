#1generate_synthetic_crop_data_sobol.py
import pandas as pd
import numpy as np
from scipy.stats.qmc import Sobol

# ========================= CONFIG =========================
N_SAMPLES_PER_CROP = 600
EXCEL_FILE = 'crop-dataset.xlsx'
RANDOM_SEED = 42
SCRAMBLE = True

def generate_sobol_synthetic_data():
    df_real = pd.read_excel(EXCEL_FILE)
    df_real.columns = [col.strip() for col in df_real.columns]

    crop_list = df_real['CROPS'].astype(str).str.strip().unique()
    synthetic_samples = []

    print(f"Generating Sobol samples ({N_SAMPLES_PER_CROP} per crop)...\n")

    for crop_idx, crop in enumerate(crop_list):
        row = df_real[df_real['CROPS'].astype(str).str.strip() == crop].iloc[0]
        print(f"→ {crop} ({crop_idx+1}/{len(crop_list)})")

        # Define mappings
        mappings = {
            'SOIL_PH': ('SOIL_PH_LOW', 'SOIL_PH_HIGH'),
            'TEMP': ('MIN_TEMP', 'MAX_TEMP'),
            'CROPDURATION': ('CROPDURATION_MIN', 'CROPDURATION_MAX'),
            'WATERREQUIRED': ('WATERREQUIRED_MIN', 'WATERREQUIRED_MAX'),
            'RELATIVE_HUMIDITY': ('RELATIVE_HUMIDITY_MIN', 'RELATIVE_HUMIDITY_MAX'),
            'N': ('N_MIN', 'N_MAX'),
            'P': ('P_MIN', 'P_MAX'),
            'K': ('K_MIN', 'K_MAX'),
        }

        feature_ranges = {}
        for feat, (min_col, max_col) in mappings.items():
            if min_col in row and max_col in row:
                a = float(row[min_col])
                b = float(row[max_col])
                if a >= b:
                    b = a + (0.1 if feat == 'SOIL_PH' else 10)
                feature_ranges[feat] = (a, b)

        features = list(feature_ranges.keys())
        if not features:
            continue

        # Generate Sobol sequence
        d = len(features)
        sampler = Sobol(d=d, scramble=SCRAMBLE, seed=RANDOM_SEED + crop_idx)
        sobol_matrix = sampler.random(n=N_SAMPLES_PER_CROP)

        # Scale to ranges
        min_vals = np.array([feature_ranges[f][0] for f in features])
        max_vals = np.array([feature_ranges[f][1] for f in features])

        scaled = min_vals + sobol_matrix * (max_vals - min_vals)

        # Create DataFrame
        crop_df = pd.DataFrame(scaled, columns=features)
        crop_df['CROPS'] = crop

        # Add categorical columns
        cat_cols = ['TYPE_OF_CROP', 'SOIL', 'SEASON', 'SOWN', 'HARVESTED', 'WATER_SOURCE']
        for col in cat_cols:
            if col in row:
                crop_df[col] = row[col]

        # Rounding
        if 'SOIL_PH' in crop_df.columns:
            crop_df['SOIL_PH'] = crop_df['SOIL_PH'].round(2)
        for col in ['TEMP', 'CROPDURATION', 'WATERREQUIRED', 'RELATIVE_HUMIDITY', 'N', 'P', 'K']:
            if col in crop_df.columns:
                crop_df[col] = crop_df[col].round(0).astype(int)

        synthetic_samples.append(crop_df)

    final_df = pd.concat(synthetic_samples, ignore_index=True)

    # Reorder columns
    final_order = ['CROPS', 'TYPE_OF_CROP', 'SOIL', 'SEASON', 'SOWN', 'HARVESTED',
                   'WATER_SOURCE', 'SOIL_PH', 'TEMP', 'CROPDURATION',
                   'WATERREQUIRED', 'RELATIVE_HUMIDITY', 'N', 'P', 'K']
    final_df = final_df[[col for col in final_order if col in final_df.columns]]

    print(f"\n✅ Sobol generation completed! Total samples: {len(final_df)}")
    return final_df


# ========================= RUN =========================
if __name__ == "__main__":
    df_synth = generate_sobol_synthetic_data()
    df_synth.to_excel('1_synthetic_crop_data_sobol.xlsx', index=False)
    print("File saved as: Synthetic_Crop_Data_Sobol_Improved.xlsx")