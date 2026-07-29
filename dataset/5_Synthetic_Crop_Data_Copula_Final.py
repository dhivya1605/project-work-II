#5generate_synthetic_crop_data_copula.py
import pandas as pd
import numpy as np
from scipy.stats import norm, truncnorm

# ========================= CONFIGURATION =========================
N_SAMPLES_PER_CROP = 600
EXCEL_FILE = 'crop-dataset.xlsx'
RANDOM_SEED = 42
CORRELATION_STRENGTH = 0.45          # Moderate correlation between features

np.random.seed(RANDOM_SEED)

def load_crop_data():
    df = pd.read_excel(EXCEL_FILE)
    df.columns = [col.strip() for col in df.columns]
    return df


def generate_copula_synthetic_data():
    original_df = load_crop_data()
    crop_list = original_df['CROPS'].astype(str).str.strip().unique()

    synthetic_samples = []

    print(f"Generating {N_SAMPLES_PER_CROP} Copula-based samples per crop...\n")

    for crop in crop_list:
        df_c = original_df[original_df['CROPS'].astype(str).str.strip() == crop].copy()
        if df_c.empty:
            continue

        # ====================== Extract Min, Max, Mean, Std per feature ======================
        feature_info = {}

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
                min_val = float(df_c[min_col].min())
                max_val = float(df_c[max_col].max())

                # Safety check
                if min_val >= max_val:
                    max_val = min_val + (0.1 if feat_name == 'SOIL_PH' else 1.0)

                mean_val = (min_val + max_val) / 2
                std_val = (max_val - min_val) / 4.0

                feature_info[feat_name] = {
                    'min': min_val,
                    'max': max_val,
                    'mean': mean_val,
                    'std': std_val
                }

        features = list(feature_info.keys())
        if len(features) < 2:
            continue

        d = len(features)

        # ====================== Gaussian Copula - Estimate Dependency ======================
        # Create correlation matrix
        corr_matrix = np.full((d, d), CORRELATION_STRENGTH)
        np.fill_diagonal(corr_matrix, 1.0)

        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            L = np.eye(d)  # Fallback to independent features

        # ====================== Generate Correlated Samples ======================
        # Draw random vector from standard normal
        Z = np.random.normal(0, 1, size=(N_SAMPLES_PER_CROP, d))

        # Apply correlation structure
        Z_correlated = Z @ L.T

        # Convert to uniform [0,1] using CDF
        U = norm.cdf(Z_correlated)

        # ====================== Scale to each feature's range ======================
        crop_data = {'CROPS': [crop] * N_SAMPLES_PER_CROP}

        for i, feat in enumerate(features):
            info = feature_info[feat]
            a, b = info['min'], info['max']
            mu, sigma = info['mean'], info['std']

            alpha = (a - mu) / sigma
            beta = (b - mu) / sigma

            # Transform uniform to values within [min, max]
            samples = truncnorm.ppf(U[:, i], alpha, beta, loc=mu, scale=sigma)
            crop_data[feat] = samples

        # Create DataFrame
        crop_df = pd.DataFrame(crop_data)

        # Add categorical columns
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
        print(f"✓ Generated {N_SAMPLES_PER_CROP} samples for: {crop}")

    # ====================== Final Dataset ======================
    final_df = pd.concat(synthetic_samples, ignore_index=True)

    final_order = ['CROPS', 'TYPE_OF_CROP', 'SOIL', 'SEASON', 'SOWN', 'HARVESTED',
                   'WATER_SOURCE', 'SOIL_PH', 'TEMP', 'CROPDURATION',
                   'WATERREQUIRED', 'RELATIVE_HUMIDITY', 'N', 'P', 'K']

    final_df = final_df[[col for col in final_order if col in final_df.columns]]

    print(f"\n✅ Copula-Based Intra-Crop Sampling Completed Successfully!")
    print(f"Total samples generated: {len(final_df)}")

    return final_df


# ========================= EXECUTION =========================
if __name__ == "__main__":
    synthetic_df = generate_copula_synthetic_data()
    output_file = '5_Synthetic_Crop_Data_Copula_Final.xlsx'
    synthetic_df.to_excel(output_file, index=False)
    print(f"\n📁 File saved: {output_file}")
    print("\nPreview:")
    print(synthetic_df.head())