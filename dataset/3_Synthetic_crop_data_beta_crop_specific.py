#3generate_synthetic_crop_data_beta.py
import pandas as pd
import numpy as np

# ========================= CONFIGURATION =========================
N_SAMPLES_PER_CROP = 600
EXCEL_FILE = 'crop-dataset.xlsx'
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def load_crop_data():
    df = pd.read_excel(EXCEL_FILE,)
    df.columns = [col.strip() for col in df.columns]
    return df


def get_beta_parameters(crop_type, feature_name):
    """
    Returns (α, β) for Beta distribution based on crop type and feature.
    """
    default = (5, 5)

    if crop_type in ['cereals', 'millets']:
        params = {
            'SOIL_PH':          (9, 6),
            'TEMP':             (7, 5),
            'CROPDURATION':     (6, 4),
            'WATERREQUIRED':    (5, 6),
            'RELATIVE_HUMIDITY':(8, 5),
            'N':                (5, 7),
            'P':                (6, 6),
            'K':                (6, 6),
        }
    elif crop_type in ['pulses']:
        params = {
            'SOIL_PH':          (8, 5),
            'TEMP':             (6, 4),
            'CROPDURATION':     (5, 3),
            'WATERREQUIRED':    (4, 5),
            'RELATIVE_HUMIDITY':(7, 4),
            'N':                (3, 8),      # Lower N for pulses
            'P':                (6, 5),
            'K':                (6, 5),
        }
    elif crop_type in ['oil seeds']:
        params = {
            'SOIL_PH':          (7, 5),
            'TEMP':             (6, 5),
            'CROPDURATION':     (5, 4),
            'WATERREQUIRED':    (5, 5),
            'RELATIVE_HUMIDITY':(6, 5),
            'N':                (5, 6),
            'P':                (5, 5),
            'K':                (5, 5),
        }
    elif crop_type in ['vegetables', 'colecrops', 'Root&tuber', 'bulbvegetables']:
        params = {
            'SOIL_PH':          (8, 6),
            'TEMP':             (8, 4),
            'CROPDURATION':     (7, 4),
            'WATERREQUIRED':    (6, 5),
            'RELATIVE_HUMIDITY':(8, 4),
            'N':                (6, 5),
            'P':                (6, 6),
            'K':                (6, 6),
        }
    else:
        params = {
            'SOIL_PH':          (8, 5),
            'TEMP':             (6, 4),
            'CROPDURATION':     (5, 4),
            'WATERREQUIRED':    (5, 5),
            'RELATIVE_HUMIDITY':(7, 4),
            'N':                (5, 6),
            'P':                (5, 5),
            'K':                (5, 5),
        }

    return params.get(feature_name, default)


def generate_beta_synthetic_data():
    original_df = load_crop_data()
    crop_list = original_df['CROPS'].astype(str).str.strip().unique()

    synthetic_samples = []

    print(f"Generating {N_SAMPLES_PER_CROP} samples per crop using Crop-Specific Beta Distribution...\n")

    for crop in crop_list:
        df_c = original_df[original_df['CROPS'].astype(str).str.strip() == crop].copy()
        if df_c.empty:
            continue

        # Get crop type for parameter selection
        crop_type = str(df_c['TYPE_OF_CROP'].iloc[0]).lower() if 'TYPE_OF_CROP' in df_c.columns else 'others'

        # ====================== IMPROVED FEATURE RANGES (Clean & Consistent) ======================
        feature_ranges = {}

        # All features defined uniformly using column name mappings
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

        # Process all features in one clean loop
        for feat_name, (min_col, max_col) in mappings.items():
            if min_col in df_c.columns and max_col in df_c.columns:
                a = float(df_c[min_col].min())
                b = float(df_c[max_col].max())

                # Safety: If min >= max, add small buffer
                if a >= b:
                    if feat_name == 'SOIL_PH':
                        b = a + 0.1
                    elif feat_name == 'TEMP':
                        b = a + 0.5
                    else:
                        b = a + 1.0

                feature_ranges[feat_name] = (a, b)

        features = list(feature_ranges.keys())
        if not features:
            continue

        # ====================== Beta Distribution Sampling ======================
        crop_data = {'CROPS': [crop] * N_SAMPLES_PER_CROP}

        for feat in features:
            a, b = feature_ranges[feat]
            alpha, beta = get_beta_parameters(crop_type, feat)

            # Sample from Beta(α, β) distribution → [0, 1]
            u = np.random.beta(alpha, beta, N_SAMPLES_PER_CROP)

            # Scale to actual [min, max] range of the crop
            samples = a + u * (b - a)

            crop_data[feat] = samples

        # Create DataFrame
        crop_df = pd.DataFrame(crop_data)

        # Add categorical columns from original data
        cat_cols = ['TYPE_OF_CROP', 'SOIL', 'SEASON', 'SOWN', 'HARVESTED', 'WATER_SOURCE']
        for col in cat_cols:
            if col in df_c.columns:
                crop_df[col] = df_c[col].iloc[0]

        # ====================== Clean Rounding ======================
        for feat in features:
            if feat == 'SOIL_PH':
                crop_df[feat] = crop_df[feat].round(2)
            else:
                crop_df[feat] = crop_df[feat].round(0).astype(int)

        synthetic_samples.append(crop_df)
        print(f"✓ Generated {N_SAMPLES_PER_CROP} samples for: {crop} ({crop_type})")

    # Combine all crops
    final_df = pd.concat(synthetic_samples, ignore_index=True)

    # Final column order
    final_order = ['CROPS', 'TYPE_OF_CROP', 'SOIL', 'SEASON', 'SOWN', 'HARVESTED',
                   'WATER_SOURCE', 'SOIL_PH', 'TEMP', 'CROPDURATION',
                   'WATERREQUIRED', 'RELATIVE_HUMIDITY', 'N', 'P', 'K']

    final_df = final_df[[col for col in final_order if col in final_df.columns]]

    print(f"\n✅ Crop-specific Beta sampling completed successfully!")
    print(f"Total samples generated: {len(final_df)}")

    return final_df


# ========================= EXECUTION =========================
if __name__ == "__main__":
    synthetic_df = generate_beta_synthetic_data()

    output_file = '3_Synthetic_crop_data_beta_crop_specific.xlsx'
    synthetic_df.to_excel(output_file, index=False)

    print(f"\n📁 File saved: {output_file}")
    print("\nPreview of first 5 rows:")
    print(synthetic_df.head())