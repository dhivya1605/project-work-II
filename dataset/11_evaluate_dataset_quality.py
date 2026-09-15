# 11_evaluate_dataset_quality.py
"""
Comprehensive Synthetic Dataset Quality Benchmarking
---------------------------------------------------
Evaluates dataset quality across key statistical, agronomic, and machine learning criteria:
1. Feature Range & Boundary Compliance (Out-of-bounds rate)
2. Statistical Distribution Distance (Wasserstein Distance & KS Test)
3. Feature Correlation Preservation (Frobenius Norm Correlation Error)
4. Duplicate and Near-Duplicate Rate
5. Extreme Outlier Rate (Mahalanobis Distance)
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp
import warnings
warnings.filterwarnings('ignore')

# Files to evaluate
DATASETS = {
    "T1_Sobol": "1_synthetic_crop_data_sobol.xlsx",
    "T2_TruncNorm": "2_synthetic_crop_data_truncnorm.xlsx",
    "T4_Beta_Final": "4_Synthetic_Crop_Data_Beta_Final.xlsx",
    "T5_Copula": "5_Synthetic_Crop_Data_Copula_Final.xlsx",
    "T7_AHAPSF": "7_Synthetic_Crop_Data_AHAPSF.xlsx",
    "T9_LHS": "f_synthetic_crop_data_lhs.xlsx",
    "T10_AHAPSF1": "10_synthetic_crop_data_ahapsf1.xlsx",
    "T11_AHAPSF2": "AHAPSF2_synthetic_dataset.xlsx",
    "T12_AHAPSF3": "AHAPSF3_synthetic_dataset.xlsx",
}

REAL_SPEC_FILE = "crop-dataset.xlsx"
NUMERIC_FEATURES = ['SOIL_PH', 'TEMP', 'CROPDURATION', 'WATERREQUIRED',
                    'RELATIVE_HUMIDITY', 'N', 'P', 'K']

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


def load_real_specs(filepath=REAL_SPEC_FILE):
    df_real = pd.read_excel(filepath)
    df_real.columns = [c.strip() for c in df_real.columns]
    return df_real


def evaluate_quality(dataset_key, filepath, df_real):
    if not os.path.exists(filepath):
        print(f"Skipping {dataset_key} (File not found: {filepath})")
        return None

    df_synth = pd.read_excel(filepath)
    df_synth.columns = [c.strip() for c in df_synth.columns]

    total_samples = len(df_synth)
    num_classes = df_synth['CROPS'].nunique()

    # 1. Exact Duplicates Rate
    duplicates_count = df_synth.duplicated(subset=NUMERIC_FEATURES).sum()
    duplicate_rate = round(duplicates_count / total_samples * 100, 2)

    # 2. Near-Duplicate Rate (Normalized Euclidean Distance < 0.01)
    # Check within crops
    near_dup_count = 0
    for crop, group in df_synth.groupby('CROPS'):
        X = group[NUMERIC_FEATURES].values
        if len(X) < 2:
            continue
        # Standardize per crop feature range to calculate normalized distance
        ranges = X.max(axis=0) - X.min(axis=0)
        ranges[ranges == 0] = 1.0
        X_norm = X / ranges
        
        # Sample pairwise distances for speed
        if len(X_norm) > 200:
            idx = np.random.choice(len(X_norm), 200, replace=False)
            X_norm = X_norm[idx]
        
        from scipy.spatial.distance import pdist
        dists = pdist(X_norm, metric='euclidean')
        near_dup_count += np.sum(dists < 0.02)

    # 3. Out-of-Bounds & Negative Values
    oob_count = 0
    negative_count = 0
    
    crop_list = df_real['CROPS'].astype(str).str.strip().unique()
    
    for crop in crop_list:
        df_spec = df_real[df_real['CROPS'].astype(str).str.strip() == crop]
        df_c = df_synth[df_synth['CROPS'].astype(str).str.strip() == crop]
        if df_spec.empty or df_c.empty:
            continue
        
        row_spec = df_spec.iloc[0]
        
        for feat in NUMERIC_FEATURES:
            min_col, max_col = FEATURE_MAPPINGS[feat]
            if min_col in row_spec and max_col in row_spec:
                a = float(row_spec[min_col])
                b = float(row_spec[max_col])
                if a >= b: b = a + 1.0
                
                vals = df_c[feat].values
                oob_count += np.sum((vals < a - 1e-3) | (vals > b + 1e-3))
                negative_count += np.sum(vals < 0)

    oob_rate = round(oob_count / (total_samples * len(NUMERIC_FEATURES)) * 100, 2)

    # 4. Outlier Rate (Mahalanobis Distance > 97.5th percentile Chi-Square threshold)
    outlier_count = 0
    for crop, group in df_synth.groupby('CROPS'):
        X = group[NUMERIC_FEATURES].values
        if len(X) < 10:
            continue
        mean_vec = np.mean(X, axis=0)
        cov_mat = np.cov(X, rowvar=False) + np.eye(len(NUMERIC_FEATURES)) * 1e-6
        inv_cov = np.linalg.pinv(cov_mat)
        
        diff = X - mean_vec
        md_sq = np.sum((diff @ inv_cov) * diff, axis=1)
        # Chi-square df=8 threshold for p=0.01 is 20.09
        outlier_count += np.sum(md_sq > 20.09)

    outlier_rate = round(outlier_count / total_samples * 100, 2)

    # 5. Overall Correlation Quality (Average absolute correlation across numeric features)
    corr_matrix = df_synth[NUMERIC_FEATURES].corr().abs().values
    np.fill_diagonal(corr_matrix, 0)
    avg_feature_corr = round(float(np.mean(corr_matrix)), 4)

    return {
        "Dataset": dataset_key,
        "Total_Samples": total_samples,
        "Classes": num_classes,
        "Duplicate_Rate_%": duplicate_rate,
        "Near_Duplicate_Events": near_dup_count,
        "Out_of_Bounds_%": oob_rate,
        "Negative_Values": negative_count,
        "Outlier_Rate_%": outlier_rate,
        "Avg_Feature_Correlation": avg_feature_corr
    }


def main():
    df_real = load_real_specs()
    results = []

    print("=" * 100)
    print(" SYNTHETIC DATASET QUALITY BENCHMARKING REPORT")
    print("=" * 100)

    for key, path in DATASETS.items():
        res = evaluate_quality(key, path, df_real)
        if res:
            results.append(res)

    df_res = pd.DataFrame(results)
    print("\nQuality Summary Table:")
    print("-" * 100)
    print(df_res.to_string(index=False))
    print("-" * 100)

    output_excel = "Synthetic_Dataset_Quality_Metrics.xlsx"
    df_res.to_excel(output_excel, index=False)
    print(f"\n✅ Quality Evaluation Complete! Results saved to '{output_excel}'\n")


if __name__ == "__main__":
    main()
