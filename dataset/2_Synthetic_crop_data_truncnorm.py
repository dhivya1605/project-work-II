import pandas as pd
import numpy as np
from scipy.stats import truncnorm

# ========================= CONFIGURATION =========================
N_SAMPLES_PER_CROP = 600
EXCEL_FILE = 'crop-dataset.xlsx'
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def load_crop_data():
    df = pd.read_excel(EXCEL_FILE)
    df.columns = [col.strip() for col in df.columns]
    return df

def generate_truncated_normal_data():
    original_df = load_crop_data()
    crop_list = original_df['CROPS'].astype(str).str.strip().unique()

    synthetic_samples = []

    print(f"Generating {N_SAMPLES_PER_CROP} samples per crop using Truncated Normal Distribution...\n")

    for crop in crop_list:
        df_c = original_df[original_df['CROPS'].astype(str).str.strip() == crop].copy()
        if df_c.empty:
            continue

        # ====================== Define Feature Ranges ======================
        feature_ranges = {}
        mappings = {
            'SOIL_PH':          ('SOIL_PH_LOW', 'SOIL_PH_HIGH'),
            'TEMP':             ('MIN_TEMP', 'MAX_TEMP'),
            'CROPDURATION':     ('CROPDURATION_MIN', 'CROPDURATION_MAX'),
            'WATERREQUIRED':    ('WATERREQUIRED_MIN', 'WATERREQUIRED_MAX'),
            'RELATIVE_HUMIDITY':('RELATIVE_HUMIDITY_MIN', 'RELATIVE_HUMIDITY_MAX'),
            'N':                ('N_MIN', 'N_MAX'),
            'P':                ('P_MIN', 'P_MAX'),
            'K':                ('K_MIN', 'K_MAX'),
        }

        for feat_name, (min_col, max_col) in mappings.items():
            if min_col in df_c.columns and max_col in df_c.columns:
                a = float(df_c[min_col].min())
                b = float(df_c[max_col].max())
                if pd.isna(a) or pd.isna(b):
                    continue
                if a == b:
                    b = a + 1.0
                feature_ranges[feat_name] = (a, b)

        features = list(feature_ranges.keys())
        if not features:
            continue

        # ====================== Generate Truncated Normal Samples ======================
        crop_data = {'CROPS': [crop] * N_SAMPLES_PER_CROP}

        for feat in features:
            a, b = feature_ranges[feat]
            mu = (a + b) / 2
            sigma = (b - a) / 4   # Reasonable spread: most values within ~±2σ

            # Truncated normal parameters
            alpha = (a - mu) / sigma
            beta = (b - mu) / sigma

            # Generate samples using scipy truncnorm
            samples = truncnorm.rvs(alpha, beta, loc=mu, scale=sigma, size=N_SAMPLES_PER_CROP)

            crop_data[feat] = samples

        # Create DataFrame for this crop
        crop_df = pd.DataFrame(crop_data)

        # Add categorical columns
        cat_cols = ['TYPE_OF_CROP', 'SOIL', 'SEASON', 'SOWN', 'HARVESTED', 'WATER_SOURCE']
        for col in cat_cols:
            if col in df_c.columns:
                crop_df[col] = df_c[col].iloc[0]

        # ====================== Clean Rounding ======================
        for col in features:
            if col == 'SOIL_PH':
                crop_df[col] = crop_df[col].round(2)
            else:
                crop_df[col] = crop_df[col].round(0).astype(int)

        synthetic_samples.append(crop_df)
        print(f"✓ Generated {N_SAMPLES_PER_CROP} samples for: {crop}")

    # Combine all crops
    final_df = pd.concat(synthetic_samples, ignore_index=True)

    # Final column order
    final_order = ['CROPS', 'TYPE_OF_CROP', 'SOIL', 'SEASON', 'SOWN', 'HARVESTED',
                   'WATER_SOURCE', 'SOIL_PH', 'TEMP', 'CROPDURATION',
                   'WATERREQUIRED', 'RELATIVE_HUMIDITY', 'N', 'P', 'K']

    final_df = final_df[[col for col in final_order if col in final_df.columns]]

    print(f"\n✅ Truncated Normal Sampling completed!")
    print(f"Total samples generated: {len(final_df)}")

    return final_df


# ========================= EXECUTION =========================
if __name__ == "__main__":
    synthetic_df = generate_truncated_normal_data()

    output_file = '2_synthetic_crop_data_truncnorm.xlsx'
    synthetic_df.to_excel(output_file, index=False)

    print(f"\n📁 File saved successfully: {output_file}")
    print("\nPreview:")
    print(synthetic_df.head())