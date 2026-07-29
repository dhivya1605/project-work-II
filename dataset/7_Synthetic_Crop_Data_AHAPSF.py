#7_Synthetic_Crop_Data_AHAPSF.xlsx
import pandas as pd
import numpy as np
from scipy.stats import qmc, truncnorm, norm

# ========================= CONFIGURATION =========================
N_SAMPLES_PER_CROP = 600
EXCEL_FILE = 'crop-dataset.xlsx'
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def load_crop_data():
    df = pd.read_excel(EXCEL_FILE)
    df.columns = [col.strip() for col in df.columns]
    return df


def generate_ahapsf_synthetic_data():
    original_df = load_crop_data()
    crop_list = original_df['CROPS'].astype(str).str.strip().unique()

    synthetic_samples = []

    print(f"Generating {N_SAMPLES_PER_CROP} samples per crop using AHAPSF (Hybrid Sampling)...\n")

    for crop in crop_list:
        df_c = original_df[original_df['CROPS'].astype(str).str.strip() == crop].copy()
        if df_c.empty:
            continue

        # ====================== Extract Min-Max Ranges ======================
        feature_ranges = {}

        mappings = {
            'SOIL_PH':          ('SOIL_PH_LOW', 'SOIL_PH_HIGH'),
            'CROPDURATION':     ('CROPDURATION_MIN', 'CROPDURATION_MAX'),
            'TEMP':             ('MIN_TEMP', 'MAX_TEMP'),
            'RELATIVE_HUMIDITY':('RELATIVE_HUMIDITY_MIN', 'RELATIVE_HUMIDITY_MAX'),
            'WATERREQUIRED':    ('WATERREQUIRED_MIN', 'WATERREQUIRED_MAX'),
            'N':                ('N_MIN', 'N_MAX'),
            'P':                ('P_MIN', 'P_MAX'),
            'K':                ('K_MIN', 'K_MAX'),
        }

        for feat, (min_col, max_col) in mappings.items():
            if min_col in df_c.columns and max_col in df_c.columns:
                a = float(df_c[min_col].min())
                b = float(df_c[max_col].max())
                if a >= b:
                    b = a + (0.1 if feat == 'SOIL_PH' else 1.0)
                feature_ranges[feat] = (a, b)

        # ====================== Hybrid Sampling for each sample ======================
        crop_data = {'CROPS': [crop] * N_SAMPLES_PER_CROP}

        n = N_SAMPLES_PER_CROP

        # 1. Soil pH → Sobol Sampling (good uniformity)
        if 'SOIL_PH' in feature_ranges:
            a, b = feature_ranges['SOIL_PH']
            sampler = qmc.Sobol(d=1, scramble=True, seed=RANDOM_SEED)
            s = sampler.random(n=n).flatten()
            crop_data['SOIL_PH'] = a + s * (b - a)

        # 2. Crop Duration → Latin Hypercube Sampling
        if 'CROPDURATION' in feature_ranges:
            a, b = feature_ranges['CROPDURATION']
            sampler = qmc.LatinHypercube(d=1, seed=RANDOM_SEED)
            u = sampler.random(n=n).flatten()
            duration_samples = a + u * (b - a)
            # Shuffle to remove ordering bias
            np.random.shuffle(duration_samples)
            crop_data['CROPDURATION'] = duration_samples

        # 3. Temperature & Relative Humidity → Truncated Normal
        for feat in ['TEMP', 'RELATIVE_HUMIDITY']:
            if feat in feature_ranges:
                a, b = feature_ranges[feat]
                mu = (a + b) / 2
                sigma = (b - a) / 4
                alpha = (a - mu) / sigma
                beta = (b - mu) / sigma
                samples = truncnorm.rvs(alpha, beta, loc=mu, scale=sigma, size=n, random_state=RANDOM_SEED)
                crop_data[feat] = samples

        # 4. Water Requirement → Beta Distribution
        if 'WATERREQUIRED' in feature_ranges:
            a, b = feature_ranges['WATERREQUIRED']
            alpha, beta = 5, 5   # Symmetric moderate spread
            u = np.random.beta(alpha, beta, n)
            crop_data['WATERREQUIRED'] = a + u * (b - a)

        # 5. N, P, K → Gaussian Copula (captures nutrient correlation)
        npk_features = ['N', 'P', 'K']
        if all(f in feature_ranges for f in npk_features):
            d = 3
            # Moderate positive correlation between N, P, K
            corr = np.full((d, d), 0.5)
            np.fill_diagonal(corr, 1.0)
            L = np.linalg.cholesky(corr)

            Z = np.random.normal(0, 1, size=(n, d))
            Z_cor = Z @ L.T
            U = norm.cdf(Z_cor)

            for i, feat in enumerate(npk_features):
                a, b = feature_ranges[feat]
                mu = (a + b) / 2
                sigma = (b - a) / 4
                alpha = (a - mu) / sigma
                beta = (b - mu) / sigma
                samples = truncnorm.ppf(U[:, i], alpha, beta, loc=mu, scale=sigma)
                crop_data[feat] = samples

        # Create DataFrame for this crop
        crop_df = pd.DataFrame(crop_data)

        # Add categorical columns
        cat_cols = ['TYPE_OF_CROP', 'SOIL', 'SEASON', 'SOWN', 'HARVESTED', 'WATER_SOURCE']
        for col in cat_cols:
            if col in df_c.columns:
                crop_df[col] = df_c[col].iloc[0]

        # ====================== Final Clean Rounding ======================
        for col in crop_df.columns:
            if col == 'SOIL_PH':
                crop_df[col] = crop_df[col].round(2)
            elif col in ['N', 'P', 'K', 'TEMP', 'CROPDURATION', 'WATERREQUIRED', 'RELATIVE_HUMIDITY']:
                crop_df[col] = crop_df[col].round(0).astype(int)

        synthetic_samples.append(crop_df)
        print(f"✓ Generated {N_SAMPLES_PER_CROP} hybrid samples for: {crop}")

    # Final dataset
    final_df = pd.concat(synthetic_samples, ignore_index=True)

    final_order = ['CROPS', 'TYPE_OF_CROP', 'SOIL', 'SEASON', 'SOWN', 'HARVESTED',
                   'WATER_SOURCE', 'SOIL_PH', 'TEMP', 'CROPDURATION',
                   'WATERREQUIRED', 'RELATIVE_HUMIDITY', 'N', 'P', 'K']

    final_df = final_df[[col for col in final_order if col in final_df.columns]]

    print(f"\n✅ AHAPSF Hybrid Sampling completed successfully!")
    print(f"Total synthetic samples: {len(final_df)}")

    return final_df


# ========================= EXECUTION =========================
if __name__ == "__main__":
    synthetic_df = generate_ahapsf_synthetic_data()

    output_file = '7_Synthetic_Crop_Data_AHAPSF.xlsx'
    synthetic_df.to_excel(output_file, index=False)

    print(f"\n📁 File saved: {output_file}")
    print("\nPreview:")
    print(synthetic_df.head())