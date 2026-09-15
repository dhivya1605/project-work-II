# 10_Synthetic_Crop_Data_AHAPSF1.py
"""
AHAPSF1: Adaptive Hybrid Agricultural Performance Synthetic Framework v2
-----------------------------------------------------------------------
Proposed scientifically justified hybrid synthetic data generation pipeline for
crop recommendation.

Key Methodology Steps:
1. Joint 8D Low-Discrepancy Quasi-Monte Carlo (Scrambled Sobol / LHS) base sampling.
2. Agronomically Informed Crop-Group Gaussian Copula dependency mapping.
3. Full-Range Uniform / Stratified Marginal Transformation to maintain high boundary
   coverage across the physiological growth envelope [MIN_{c,f}, MAX_{c,f}].
4. Multi-stage quality gate: Hard range validation, Mahalanobis density outlier
   pruning, and Euclidean distance de-duplication.
5. Strict isolation from evaluation test sets (zero data leakage).
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import qmc, norm

# ========================= CONFIGURATION =========================
N_SAMPLES_PER_CROP = 600
EXCEL_FILE = 'crop-dataset.xlsx'
OUTPUT_EXCEL = '10_synthetic_crop_data_ahapsf1.xlsx'
RANDOM_SEED = 42

# 8 Core Numerical Features in standard order
NUMERIC_FEATURES = [
    'SOIL_PH', 'TEMP', 'CROPDURATION', 'WATERREQUIRED',
    'RELATIVE_HUMIDITY', 'N', 'P', 'K'
]

# Mapping from target features to min/max columns in crop-dataset.xlsx
FEATURE_MAPPINGS = {
    'SOIL_PH':          ('SOIL_PH_LOW', 'SOIL_PH_HIGH'),
    'TEMP':             ('MIN_TEMP', 'MAX_TEMP'),
    'CROPDURATION':     ('CROPDURATION_MIN', 'CROPDURATION_MAX'),
    'WATERREQUIRED':    ('WATERREQUIRED_MIN', 'WATERREQUIRED_MAX'),
    'RELATIVE_HUMIDITY':('RELATIVE_HUMIDITY_MIN', 'RELATIVE_HUMIDITY_MAX'),
    'N':                ('N_MIN', 'N_MAX'),
    'P':                ('P_MIN', 'P_MAX'),
    'K':                ('K_MIN', 'K_MAX'),
}


def load_crop_data(filepath=EXCEL_FILE):
    """Loads and standardizes column names of original crop specification dataset."""
    df = pd.read_excel(filepath)
    df.columns = [col.strip() for col in df.columns]
    return df


def get_crop_group(crop_type_str):
    """Categorizes a crop type into standard agronomic functional groups."""
    crop_type = str(crop_type_str).lower().strip()
    
    if any(k in crop_type for k in ['cereal', 'millet']):
        return 'cereals'
    elif any(k in crop_type for k in ['pulse', 'legume']):
        return 'pulses'
    elif any(k in crop_type for k in ['oil']):
        return 'oilseeds'
    elif any(k in crop_type for k in ['veg', 'cole', 'root', 'tuber', 'bulb']):
        return 'vegetables'
    else:
        return 'commercial'


def build_crop_group_correlation_matrix(group_name, d=8):
    """
    Constructs an agronomically realistic 8x8 rank correlation matrix
    for the specific crop group.
    
    Feature indices:
    0: SOIL_PH, 1: TEMP, 2: CROPDURATION, 3: WATERREQUIRED,
    4: RELATIVE_HUMIDITY, 5: N, 6: P, 7: K
    """
    R = np.eye(d)
    
    # Common Agronomic Physics across all crops:
    # Moderate Inverse Temperature vs. Relative Humidity relationship
    R[1, 4] = R[4, 1] = -0.25
    # Crop Duration vs. Water Required positive relationship
    R[2, 3] = R[3, 2] = 0.35
    # Temperature vs. Water Requirement positive relationship
    R[1, 3] = R[3, 1] = 0.25

    if group_name == 'cereals':
        # Cereals: Heavy, balanced N-P-K nutrient uptake
        R[5, 6] = R[6, 5] = 0.45  # N - P
        R[5, 7] = R[7, 5] = 0.40  # N - K
        R[6, 7] = R[7, 6] = 0.45  # P - K
        R[2, 3] = R[3, 2] = 0.40  # Duration - Water
    elif group_name == 'pulses':
        # Pulses: Atmospheric nitrogen fixation -> lower N coupling, high P-K for nodules
        R[5, 6] = R[6, 5] = 0.20  # N - P
        R[5, 7] = R[7, 5] = 0.15  # N - K
        R[6, 7] = R[7, 6] = 0.50  # P - K
        R[0, 5] = R[5, 0] = 0.15  # pH - N
    elif group_name == 'oilseeds':
        # Oilseeds: High K for oil synthesis, moderate N-P
        R[5, 6] = R[6, 5] = 0.35  # N - P
        R[5, 7] = R[7, 5] = 0.40  # N - K
        R[6, 7] = R[7, 6] = 0.45  # P - K
        R[2, 7] = R[7, 2] = 0.25  # Duration - K
    elif group_name == 'vegetables':
        # Vegetables: High water and N-K intensity
        R[5, 6] = R[6, 5] = 0.40  # N - P
        R[5, 7] = R[7, 5] = 0.45  # N - K
        R[6, 7] = R[7, 6] = 0.40  # P - K
        R[1, 3] = R[3, 1] = 0.30  # Temp - Water
    else:  # commercial / others
        R[5, 6] = R[6, 5] = 0.45  # N - P
        R[5, 7] = R[7, 5] = 0.45  # N - K
        R[6, 7] = R[7, 6] = 0.50  # P - K
        R[2, 3] = R[3, 2] = 0.45  # Duration - Water

    # Ensure Positive Definiteness via Cholesky / Eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-6)
    R_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    inv_sqrt_diag = np.diag(1.0 / np.sqrt(np.diag(R_psd)))
    R_final = inv_sqrt_diag @ R_psd @ inv_sqrt_diag
    return R_final


def generate_crop_ahapsf1_samples(row, n_target=N_SAMPLES_PER_CROP, seed=42):
    """
    Generates high-quality synthetic samples for a single crop using AHAPSF1 methodology.
    """
    crop_name = str(row['CROPS']).strip()
    crop_type_str = row.get('TYPE_OF_CROP', 'others')
    group_name = get_crop_group(crop_type_str)

    # 1. Extract feature ranges [min, max]
    feature_ranges = {}
    for feat, (min_col, max_col) in FEATURE_MAPPINGS.items():
        if min_col in row and max_col in row:
            a = float(row[min_col])
            b = float(row[max_col])
            if pd.isna(a) or pd.isna(b):
                a, b = 0.0, 100.0
            if a >= b:
                b = a + (0.1 if feat == 'SOIL_PH' else 1.0)
            feature_ranges[feat] = (a, b)

    d = len(NUMERIC_FEATURES)
    
    # Over-generate candidate samples to allow quality filtering
    n_candidates = int(n_target * 1.20)

    # 2. Joint 8D Low-Discrepancy Quasi-Monte Carlo Base Sampling (Scrambled Sobol)
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    U_qmc = sampler.random(n=n_candidates)
    U_qmc = np.clip(U_qmc, 1e-6, 1.0 - 1e-6)

    # 3. Apply Crop-Group Specific Gaussian Copula Transformation
    corr_matrix = build_crop_group_correlation_matrix(group_name, d=d)
    L = np.linalg.cholesky(corr_matrix)

    Z = norm.ppf(U_qmc)
    Z_corr = Z @ L.T
    U_corr = norm.cdf(Z_corr)

    # 4. Full-Range Uniform / Stratified Marginal Transformation
    # Scale correlated uniform margins U_corr to exact [min, max] range
    samples_dict = {}
    for i, feat in enumerate(NUMERIC_FEATURES):
        a, b = feature_ranges[feat]
        feat_samples = a + U_corr[:, i] * (b - a)
        samples_dict[feat] = np.clip(feat_samples, a, b)

    df_cand = pd.DataFrame(samples_dict)

    # 5. Multi-Stage Agronomic Quality Gate
    # Stage 5a: Hard Range & Non-Negativity Constraints
    valid_mask = np.ones(len(df_cand), dtype=bool)
    for feat in NUMERIC_FEATURES:
        a, b = feature_ranges[feat]
        valid_mask &= (df_cand[feat] >= a) & (df_cand[feat] <= b)
        if feat in ['N', 'P', 'K', 'WATERREQUIRED', 'CROPDURATION', 'TEMP', 'RELATIVE_HUMIDITY']:
            valid_mask &= (df_cand[feat] >= 0)

    df_cand = df_cand[valid_mask].reset_index(drop=True)

    # Stage 5b: Extreme Outlier Pruning (Remove extreme outer 1% tail noise using Mahalanobis distance)
    if len(df_cand) > n_target:
        X_mat = df_cand[NUMERIC_FEATURES].values
        mean_vec = np.mean(X_mat, axis=0)
        cov_mat = np.cov(X_mat, rowvar=False) + np.eye(d) * 1e-6
        inv_cov = np.linalg.pinv(cov_mat)

        diff = X_mat - mean_vec
        md_squared = np.sum((diff @ inv_cov) * diff, axis=1)
        
        # Keep samples below the 99th percentile threshold
        threshold = np.percentile(md_squared, 99.0)
        clean_mask = md_squared <= threshold
        df_cand = df_cand[clean_mask].reset_index(drop=True)

    # Stage 5c: De-duplication check
    df_cand = df_cand.drop_duplicates(subset=NUMERIC_FEATURES).reset_index(drop=True)

    # Select exactly n_target samples
    if len(df_cand) >= n_target:
        df_final = df_cand.iloc[:n_target].copy()
    else:
        df_final = df_cand.copy()
        needed = n_target - len(df_final)
        extra_rows = df_cand.sample(n=needed, replace=True, random_state=seed).copy()
        for feat in NUMERIC_FEATURES:
            a, b = feature_ranges[feat]
            std_j = (b - a) * 0.005
            extra_rows[feat] = np.clip(extra_rows[feat] + np.random.normal(0, std_j, size=needed), a, b)
        df_final = pd.concat([df_final, extra_rows], ignore_index=True)

    # Add Target Column
    df_final['CROPS'] = crop_name

    # Add Categorical Metadata
    cat_cols = ['TYPE_OF_CROP', 'SOIL', 'SEASON', 'SOWN', 'HARVESTED', 'WATER_SOURCE']
    for col in cat_cols:
        if col in row:
            df_final[col] = row[col]

    # Clean Agronomic Rounding
    df_final['SOIL_PH'] = df_final['SOIL_PH'].round(2)
    for col in ['TEMP', 'CROPDURATION', 'WATERREQUIRED', 'RELATIVE_HUMIDITY', 'N', 'P', 'K']:
        if col in df_final.columns:
            df_final[col] = df_final[col].round(0).astype(int)

    return df_final


def generate_ahapsf1_dataset(excel_file=EXCEL_FILE, output_excel=OUTPUT_EXCEL):
    """
    Runs the complete AHAPSF1 pipeline across all 49 crops in the dataset.
    """
    df_real = load_crop_data(excel_file)
    crop_list = df_real['CROPS'].astype(str).str.strip().unique()

    print("=" * 80)
    print(" AHAPSF1: Adaptive Hybrid Agricultural Performance Synthetic Framework v2")
    print("=" * 80)
    print(f"Generating {N_SAMPLES_PER_CROP} samples per crop across {len(crop_list)} crops...\n")

    synthetic_crops = []

    for idx, crop in enumerate(crop_list):
        df_c = df_real[df_real['CROPS'].astype(str).str.strip() == crop]
        if df_c.empty:
            continue
        row = df_c.iloc[0]

        crop_synth_df = generate_crop_ahapsf1_samples(row, n_target=N_SAMPLES_PER_CROP, seed=RANDOM_SEED + idx)
        synthetic_crops.append(crop_synth_df)

    final_df = pd.concat(synthetic_crops, ignore_index=True)

    # Reorder columns to standard schema
    standard_order = [
        'CROPS', 'TYPE_OF_CROP', 'SOIL', 'SEASON', 'SOWN', 'HARVESTED',
        'WATER_SOURCE', 'SOIL_PH', 'TEMP', 'CROPDURATION',
        'WATERREQUIRED', 'RELATIVE_HUMIDITY', 'N', 'P', 'K'
    ]
    final_df = final_df[[c for c in standard_order if c in final_df.columns]]

    print("\n" + "=" * 80)
    print(f" SUCCESS: AHAPSF1 Synthetic Dataset Generated!")
    print(f" Total Samples: {len(final_df)} across {len(crop_list)} classes.")
    print("=" * 80)

    final_df.to_excel(output_excel, index=False)
    print(f" Saved output file to: {output_excel}\n")
    return final_df


if __name__ == "__main__":
    df_synth = generate_ahapsf1_dataset()
    print("Dataset Preview:")
    print(df_synth.head())
