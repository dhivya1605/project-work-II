#8_Synthetic_Crop_Data_Hybrid_Final.py
import pandas as pd
import numpy as np
from scipy.stats import qmc, truncnorm, norm

# ========================= CONFIGURATION =========================
N_SAMPLES_PER_CROP = 600
EXCEL_FILE = 'crop-dataset.xlsx'
RANDOM_SEED = 42
COPULA_CORRELATION = 0.5

np.random.seed(RANDOM_SEED)

def load_crop_data():
    df = pd.read_excel(EXCEL_FILE)
    df.columns = [col.strip() for col in df.columns]
    return df


def generate_hybrid_synthetic_data():
    original_df = load_crop_data()
    crop_list = original_df['CROPS'].astype(str).str.strip().unique()

    synthetic_samples = []

    print(f"Generating {N_SAMPLES_PER_CROP} Hybrid samples per crop...\n")

    for crop in crop_list:
        df_c = original_df[original_df['CROPS'].astype(str).str.strip() == crop].copy()
        if df_c.empty:
            continue

        # ====================== Extract Ranges ======================
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

        for feat, (min_col, max_col) in mappings.items():
            if min_col in df_c.columns and max_col in df_c.columns:
                a = float(df_c[min_col].min())
                b = float(df_c[max_col].max())
                if a >= b:
                    b = a + (0.1 if feat == 'SOIL_PH' else 1.0)
                feature_ranges[feat] = (a, b)

        features = list(feature_ranges.keys())
        n = N_SAMPLES_PER_CROP
        crop_data = {'CROPS': [crop] * n}

        # ====================== HYBRID SAMPLING ======================

        # 1. SOIL_PH → Sobol (Best uniformity)
        if 'SOIL_PH' in feature_ranges:
            a, b = feature_ranges['SOIL_PH']
            sampler = qmc.Sobol(d=1, scramble=True, seed=RANDOM_SEED)
            s = sampler.random(n=n).flatten()
            crop_data['SOIL_PH'] = a + s * (b - a)

        # 2. CROPDURATION → Latin Hypercube
        if 'CROPDURATION' in feature_ranges:
            a, b = feature_ranges['CROPDURATION']
            sampler = qmc.LatinHypercube(d=1, seed=RANDOM_SEED)
            u = sampler.random(n=n).flatten()
            samples = a + u * (b - a)
            np.random.shuffle(samples)
            crop_data['CROPDURATION'] = samples

        # 3. TEMP & RELATIVE_HUMIDITY → Truncated Normal
        for feat in ['TEMP', 'RELATIVE_HUMIDITY']:
            if feat in feature_ranges:
                a, b = feature_ranges[feat]
                mu = (a + b) / 2
                sigma = (b - a) / 4
                alpha = (a - mu) / sigma
                beta = (b - mu) / sigma
                crop_data[feat] = truncnorm.rvs(alpha, beta, loc=mu, scale=sigma, size=n)

        # 4. WATERREQUIRED → Beta Distribution
        if 'WATERREQUIRED' in feature_ranges:
            a, b = feature_ranges['WATERREQUIRED']
            u = np.random.beta(5, 5, n)          # Symmetric moderate spread
            crop_data['WATERREQUIRED'] = a + u * (b - a)

        # 5. N, P, K → Gaussian Copula (Strong Correlation)
        npk = ['N', 'P', 'K']
        if all(f in feature_ranges for f in npk):
            d = 3
            corr = np.full((d, d), COPULA_CORRELATION)
            np.fill_diagonal(corr, 1.0)
            L = np.linalg.cholesky(corr)

            Z = np.random.normal(0, 1, size=(n, d))
            Zc = Z @ L.T
            U = norm.cdf(Zc)

            for i, feat in enumerate(npk):
                a, b = feature_ranges[feat]
                mu = (a + b) / 2
                sigma = (b - a) / 4
                alpha = (a - mu) / sigma
                beta = (b - mu) / sigma
                crop_data[feat] = truncnorm.ppf(U[:, i], alpha, beta, loc=mu, scale=sigma)

        # Create DataFrame
        crop_df = pd.DataFrame(crop_data)

        # Add categorical columns
        cat_cols = ['TYPE_OF_CROP', 'SOIL', 'SEASON', 'SOWN', 'HARVESTED', 'WATER_SOURCE']
        for col in cat_cols:
            if col in df_c.columns:
                crop_df[col] = df_c[col].iloc[0]

        # Clean Rounding
        for col in crop_df.select_dtypes(include=np.number).columns:
            if col == 'SOIL_PH':
                crop_df[col] = crop_df[col].round(2)
            else:
                crop_df[col] = crop_df[col].round(0).astype(int)

        synthetic_samples.append(crop_df)
        print(f"✓ Generated {n} hybrid samples for: {crop}")

    final_df = pd.concat(synthetic_samples, ignore_index=True)

    final_order = ['CROPS', 'TYPE_OF_CROP', 'SOIL', 'SEASON', 'SOWN', 'HARVESTED',
                   'WATER_SOURCE', 'SOIL_PH', 'TEMP', 'CROPDURATION',
                   'WATERREQUIRED', 'RELATIVE_HUMIDITY', 'N', 'P', 'K']

    final_df = final_df[[col for col in final_order if col in final_df.columns]]

    print(f"\n✅ Hybrid Model (HAPSM) Completed!")
    print(f"Total samples: {len(final_df)}")

    return final_df


# ========================= RUN =========================
if __name__ == "__main__":
    synthetic_df = generate_hybrid_synthetic_data()

    output_file = '8_Synthetic_Crop_Data_Hybrid_Final.xlsx'
    synthetic_df.to_excel(output_file, index=False)

    print(f"\n📁 File saved: {output_file}")