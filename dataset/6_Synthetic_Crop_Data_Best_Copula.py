#6_generate_synthetic_crop_data_hybrid.py
import pandas as pd
import numpy as np
from scipy.stats import norm, truncnorm
import warnings
warnings.filterwarnings('ignore')

# ========================= CONFIGURATION =========================
N_SAMPLES_PER_CROP = 600
EXCEL_FILE = 'crop-dataset.xlsx'
RANDOM_SEED = 42
CORRELATION_STRENGTH = 0.45   # Moderate positive correlation (best balance)

np.random.seed(RANDOM_SEED)

def load_crop_data():
    df = pd.read_excel(EXCEL_FILE)
    df.columns = [col.strip() for col in df.columns]
    return df

def generate_best_synthetic_data():
    original_df = load_crop_data()
    crop_list = original_df['CROPS'].astype(str).str.strip().unique()

    synthetic_samples = []

    print(f"Generating {N_SAMPLES_PER_CROP} samples per crop using Gaussian Copula + Truncated Normal...\n")

    for crop in crop_list:
        df_c = original_df[original_df['CROPS'].astype(str).str.strip() == crop].copy()
        if df_c.empty:
            continue

        # ====================== Feature Definition ======================
        feature_info = {}

        # SOIL PH
        if 'SOIL_PH_LOW' in df_c.columns and 'SOIL_PH_HIGH' in df_c.columns:
            a = float(df_c['SOIL_PH_LOW'].min())
            b = float(df_c['SOIL_PH_HIGH'].max())
            if a >= b: b = a + 0.05
            feature_info['SOIL_PH'] = {'min': a, 'max': b, 'mean': (a + b)/2}

        # TEMPERATURE
        if 'MIN_TEMP' in df_c.columns and 'MAX_TEMP' in df_c.columns:
            a = float(df_c['MIN_TEMP'].min())
            b = float(df_c['MAX_TEMP'].max())
            if a >= b: b = a + 0.5
            feature_info['TEMP'] = {'min': a, 'max': b, 'mean': (a + b)/2}

        # Other features
        mappings = {
            'CROPDURATION':     ('CROPDURATION_MIN', 'CROPDURATION_MAX'),
            'WATERREQUIRED':    ('WATERREQUIRED_MIN', 'WATERREQUIRED_MAX'),
            'RELATIVE_HUMIDITY':('RELATIVE_HUMIDITY_MIN', 'RELATIVE_HUMIDITY_MAX'),
            'N':                ('N_MIN', 'N_MAX'),
            'P':                ('P_MIN', 'P_MAX'),
            'K':                ('K_MIN', 'K_MAX'),
        }

        for feat, (min_col, max_col) in mappings.items():
            if min_col in df_c.columns and max_col in df_c.columns:
                a = float(df_c[min_col].min())
                b = float(df_c[max_col].max())
                if pd.isna(a) or pd.isna(b) or a >= b:
                    continue
                feature_info[feat] = {'min': a, 'max': b, 'mean': (a + b)/2}

        features = list(feature_info.keys())
        if len(features) < 2:
            continue  # Need at least 2 features for copula benefit

        d = len(features)

        # Create correlation matrix (moderate positive dependency)
        corr_matrix = np.full((d, d), CORRELATION_STRENGTH)
        np.fill_diagonal(corr_matrix, 1.0)

        try:
            L = np.linalg.cholesky(corr_matrix)
        except:
            L = np.eye(d)

        # ====================== Gaussian Copula Sampling ======================
        Z = np.random.normal(0, 1, size=(N_SAMPLES_PER_CROP, d))
        Z_correlated = Z @ L.T
        U = norm.cdf(Z_correlated)                    # Uniform [0,1]

        # Transform to truncated normal within bounds
        crop_data = {'CROPS': [crop] * N_SAMPLES_PER_CROP}

        for i, feat in enumerate(features):
            info = feature_info[feat]
            a, b = info['min'], info['max']
            mu = info['mean']
            sigma = (b - a) / 4.0                      # Good spread

            alpha = (a - mu) / sigma
            beta = (b - mu) / sigma

            samples = truncnorm.ppf(U[:, i], alpha, beta, loc=mu, scale=sigma)
            crop_data[feat] = samples

        crop_df = pd.DataFrame(crop_data)

        # Add categorical columns
        cat_cols = ['TYPE_OF_CROP', 'SOIL', 'SEASON', 'SOWN', 'HARVESTED', 'WATER_SOURCE']
        for col in cat_cols:
            if col in df_c.columns:
                crop_df[col] = df_c[col].iloc[0]

        # ====================== Final Clean Rounding ======================
        for feat in features:
            if feat == 'SOIL_PH':
                crop_df[feat] = crop_df[feat].round(2)
            else:
                crop_df[feat] = crop_df[feat].round(0).astype(int)

        synthetic_samples.append(crop_df)
        print(f"✓ Generated {N_SAMPLES_PER_CROP} samples for: {crop}")

    final_df = pd.concat(synthetic_samples, ignore_index=True)

    final_order = ['CROPS', 'TYPE_OF_CROP', 'SOIL', 'SEASON', 'SOWN', 'HARVESTED',
                   'WATER_SOURCE', 'SOIL_PH', 'TEMP', 'CROPDURATION',
                   'WATERREQUIRED', 'RELATIVE_HUMIDITY', 'N', 'P', 'K']

    final_df = final_df[[col for col in final_order if col in final_df.columns]]

    print(f"\n✅ Best method (Gaussian Copula) completed successfully!")
    print(f"Total synthetic samples: {len(final_df)}")

    return final_df


# ========================= RUN =========================
if __name__ == "__main__":
    synthetic_df = generate_best_synthetic_data()

    output_file = '6_Synthetic_Crop_Data_Best_Copula.xlsx'
    synthetic_df.to_excel(output_file, index=False)

    print(f"\n📁 File saved: {output_file}")
    print("\nFirst 5 rows:")
    print(synthetic_df.head())