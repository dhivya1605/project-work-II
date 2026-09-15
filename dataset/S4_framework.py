# S4_framework.py
"""
S4 — Joint Constraint and Boundary-Aware Synthetic Data Generation
==================================================================
A genuinely new synthetic-data generation framework for agricultural crop data.

CORE IDEA:
  Instead of generating each feature independently and concatenating,
  S4 generates, scores, and selects COMPLETE 8-dimensional feature vectors
  subject to crop-specific joint constraints.

Key methodological differences from the three existing baseline frameworks:

  Baseline-1 (10_Synthetic_Crop_Data_AHAPSF1.py):
    - Uses joint 8D Sobol + full 8-feature Gaussian copula with HARD-CODED
      crop-group correlation matrices (cereals/pulses/oilseeds/vegetables/commercial).
    - Selects final 600 as simply the FIRST N valid rows after Mahalanobis pruning.
    - Oversamples by only 1.2×; no 8D maximin diversity selection.

  Baseline-2 (7_Synthetic_Crop_Data_AHAPSF.py):
    - Generates each feature INDEPENDENTLY (Sobol/LHS/TruncNorm/Beta/Copula).
    - NPK copula uses HARD-CODED correlation=0.5 for all crops.
    - No diversity control, no duplicate detection, no quality-based selection.

  Baseline-3 (AHAPSF2_framework.py / AHAPSF3_framework.py):
    - Best adaptive per-feature sampler selection with data-driven parameters.
    - Still FEATURE-WISE concatenation; no joint 8D selection.
    - Gaussian copula applied only to NPK; other features remain independent.
    - Near-duplicate jitter but NO maximin diversity selection in joint 8D space.

S4 NOVELTY:
  1. ALL candidate generation operates in the FULL normalized [0,1]^8 space.
  2. Joint feasibility score evaluates each COMPLETE feature vector holistically.
  3. Greedy maximin diversity selection in normalized 8D space (NOT first-N selection).
  4. Conservative data-driven dependency: only preserve correlations where
     |Spearman rho| >= DEPENDENCY_THRESHOLD (no fabricated 0.5 correlations).
  5. Adaptive marginal shaping: distribution parameters derived from range statistics,
     not fixed across all crops.
  6. 3x oversampling with 4 joint-space candidate mechanisms for rich candidate pool.

IMPORTANT LIMITATIONS (honest statement):
  The source dataset is RANGE/SPECIFICATION BASED (not observational).
  S4 does NOT claim to learn real-world distributions.
  Dependency structures are estimated from 49 crop-midpoint correlations — a
  conservative but honest use of available information.
  S4 does NOT train any ML model during generation.

Authors: S4 Framework
Date: 2026
"""

import os
import sys
import math
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from scipy.stats import (
    qmc, truncnorm, beta as scipy_beta, norm,
    skew as sp_skew, kurtosis as sp_kurtosis,
    wasserstein_distance, spearmanr
)
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RANDOM_SEED = 42
N_SAMPLES_PER_CROP = 600
OVERSAMPLING_FACTOR = 3          # Generate 3x candidates before selection
DEPENDENCY_THRESHOLD = 0.25      # Only preserve Spearman |rho| >= this value
NEAR_DUPLICATE_THRESHOLD = 0.02  # Euclidean distance in normalized 8D space
JITTER_SCALE = 0.003             # Max jitter fraction of range (pre-final-round only)
BOUNDARY_TOL = 1e-4              # Tolerance for boundary validation

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_EXCEL = os.path.join(DATASET_DIR, 'crop-dataset.xlsx')

OUTPUT_SYNTHETIC_EXCEL    = os.path.join(DATASET_DIR, 'S4_synthetic_dataset.xlsx')
OUTPUT_FEATURE_ANALYSIS   = os.path.join(DATASET_DIR, 'S4_feature_analysis.xlsx')
OUTPUT_DEPENDENCY_ANALYSIS= os.path.join(DATASET_DIR, 'S4_dependency_analysis.xlsx')
OUTPUT_CANDIDATE_COMP     = os.path.join(DATASET_DIR, 'S4_candidate_comparison.xlsx')
OUTPUT_JOINT_SPACE        = os.path.join(DATASET_DIR, 'S4_joint_space_analysis.xlsx')
OUTPUT_QUALITY_ANALYSIS   = os.path.join(DATASET_DIR, 'S4_quality_analysis.xlsx')
OUTPUT_GEN_SUMMARY        = os.path.join(DATASET_DIR, 'S4_generation_summary.xlsx')
OUTPUT_CONCLUSION_REPORT  = os.path.join(DATASET_DIR, 'S4_conclusion_report.md')

NUMERIC_FEATURES = [
    'SOIL_PH', 'TEMP', 'CROPDURATION', 'WATERREQUIRED',
    'RELATIVE_HUMIDITY', 'N', 'P', 'K'
]
D = len(NUMERIC_FEATURES)  # 8

CATEGORICAL_FEATURES = [
    'TYPE_OF_CROP', 'SOIL', 'SEASON', 'SOWN', 'HARVESTED', 'WATER_SOURCE'
]

FEATURE_MAPPINGS = {
    'SOIL_PH':           ('SOIL_PH_LOW',          'SOIL_PH_HIGH'),
    'TEMP':              ('MIN_TEMP',              'MAX_TEMP'),
    'CROPDURATION':      ('CROPDURATION_MIN',      'CROPDURATION_MAX'),
    'WATERREQUIRED':     ('WATERREQUIRED_MIN',     'WATERREQUIRED_MAX'),
    'RELATIVE_HUMIDITY': ('RELATIVE_HUMIDITY_MIN', 'RELATIVE_HUMIDITY_MAX'),
    'N':                 ('N_MIN',                 'N_MAX'),
    'P':                 ('P_MIN',                 'P_MAX'),
    'K':                 ('K_MIN',                 'K_MAX'),
}

# Fix global random seed
np.random.seed(RANDOM_SEED)
rng_global = np.random.RandomState(RANDOM_SEED)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})


# ==============================================================================
# STEP 1: LOAD SOURCE SPECIFICATION
# ==============================================================================
def load_source_specification(filepath=INPUT_EXCEL):
    """
    Loads and cleans the crop specification dataset.
    The source is RANGE-BASED (min/max per crop), not observational samples.
    """
    for path in [filepath, 'crop-dataset.xlsx', os.path.join('dataset', 'crop-dataset.xlsx')]:
        if os.path.exists(path):
            filepath = path
            break
    else:
        raise FileNotFoundError(f"crop-dataset.xlsx not found near {DATASET_DIR}")

    df = pd.read_excel(filepath)
    df.columns = [col.strip() for col in df.columns]

    if 'CROPS' not in df.columns:
        raise ValueError("Column 'CROPS' not found in source specification.")

    df['CROPS'] = df['CROPS'].astype(str).str.strip().str.lower()
    print(f"[S4] Loaded source specification: {len(df)} crop rows from {filepath}")
    return df


# ==============================================================================
# STEP 2: ANALYZE CROP RANGES
# ==============================================================================
def extract_crop_ranges(df_row):
    """
    Extracts feature-wise (min, max) bounds for a single crop spec row.
    Guarantees a > b is corrected with a minimal epsilon expansion.
    Zero-width ranges receive a small safe expansion.
    """
    ranges = {}
    for feat, (min_col, max_col) in FEATURE_MAPPINGS.items():
        if min_col in df_row.index and max_col in df_row.index:
            a = float(df_row[min_col]) if pd.notna(df_row[min_col]) else 0.0
            b = float(df_row[max_col]) if pd.notna(df_row[max_col]) else 100.0
        else:
            a, b = 0.0, 100.0
        # Ensure strictly a < b
        if a >= b:
            eps = 0.1 if feat == 'SOIL_PH' else 1.0
            b = a + eps
        ranges[feat] = (a, b)
    return ranges


def analyze_crop_ranges(df_real):
    """
    Computes per-feature statistical summaries across all crop specifications.
    Returns analysis DataFrame and dict of feature midpoints + spans.
    """
    print("\n" + "=" * 80)
    print("S4 STEP 2: ANALYZING CROP RANGE SPECIFICATIONS")
    print("=" * 80)

    feature_mids   = {}
    feature_spans  = {}
    analysis_rows  = []

    for feat in NUMERIC_FEATURES:
        min_col, max_col = FEATURE_MAPPINGS[feat]
        mins  = df_real[min_col].astype(float).values
        maxs  = df_real[max_col].astype(float).values
        mids  = (mins + maxs) / 2.0
        spans = maxs - mins

        feature_mids[feat]  = mids
        feature_spans[feat] = spans

        # Per-feature statistics across crop specifications
        global_min = float(np.min(mins))
        global_max = float(np.max(maxs))
        mean_mid   = float(np.mean(mids))
        std_mid    = float(np.std(mids, ddof=1)) if len(mids) > 1 else 0.0
        mean_span  = float(np.mean(spans))
        skew_val   = float(sp_skew(mids)) if len(mids) > 2 else 0.0
        kurt_val   = float(sp_kurtosis(mids)) if len(mids) > 2 else 0.0
        rel_span   = float(np.mean(spans / np.maximum(np.abs(mids), 1e-3)))

        # Determine recommended marginal shape based on range characteristics
        if feat == 'SOIL_PH':
            recommended_marginal = 'sobol_uniform'    # narrow, continuous, needs full coverage
        elif feat == 'CROPDURATION':
            recommended_marginal = 'lhs_stratified'   # wide, needs stratification
        elif feat == 'WATERREQUIRED':
            if skew_val > 0.3:
                recommended_marginal = 'beta_right'   # right-skewed
            elif skew_val < -0.3:
                recommended_marginal = 'beta_left'
            else:
                recommended_marginal = 'beta_symmetric'
        elif feat in ['TEMP', 'RELATIVE_HUMIDITY']:
            recommended_marginal = 'truncnorm'        # physiological bell
        elif feat in ['N', 'P', 'K']:
            recommended_marginal = 'copula_coupled'   # dependency-aware
        else:
            recommended_marginal = 'lhs_stratified'

        analysis_rows.append({
            'Feature':              feat,
            'Global_Min':           round(global_min, 3),
            'Global_Max':           round(global_max, 3),
            'Midpoint_Mean':        round(mean_mid, 3),
            'Midpoint_Std':         round(std_mid, 3),
            'Midpoint_Skewness':    round(skew_val, 4),
            'Midpoint_Kurtosis':    round(kurt_val, 4),
            'Avg_Span_Width':       round(mean_span, 3),
            'Relative_Span_Ratio':  round(rel_span, 4),
            'N_Crops':              len(mids),
            'Recommended_Marginal': recommended_marginal,
        })

    df_analysis = pd.DataFrame(analysis_rows)
    print(df_analysis[['Feature', 'Global_Min', 'Global_Max', 'Midpoint_Skewness',
                        'Avg_Span_Width', 'Recommended_Marginal']].to_string(index=False))
    return df_analysis, feature_mids, feature_spans


# ==============================================================================
# STEP 3: NORMALIZE FEATURES (per-crop to [0,1])
# ==============================================================================
def normalize_features(X_raw, ranges):
    """
    Normalizes raw feature matrix (n_candidates x 8) to [0,1] per feature
    using crop-specific bounds. Prevents large-scale features from dominating
    distance calculations (WATERREQUIRED, N, P, K vs SOIL_PH).

    x_norm[i,j] = (x[i,j] - min_j) / (max_j - min_j)
    Handles zero-width ranges safely.
    """
    X_norm = np.zeros_like(X_raw, dtype=float)
    for j, feat in enumerate(NUMERIC_FEATURES):
        a, b = ranges[feat]
        span = max(b - a, 1e-8)
        X_norm[:, j] = (X_raw[:, j] - a) / span
    return X_norm


def denormalize_features(X_norm, ranges):
    """Converts normalized [0,1]^8 back to original feature scales."""
    X_raw = np.zeros_like(X_norm, dtype=float)
    for j, feat in enumerate(NUMERIC_FEATURES):
        a, b = ranges[feat]
        X_raw[:, j] = a + X_norm[:, j] * (b - a)
    return X_raw


# ==============================================================================
# STEP 4: BUILD DEPENDENCY STRUCTURE (data-driven, conservative)
# ==============================================================================
def build_dependency_structure(df_real, feature_mids):
    """
    Constructs a data-driven, conservative 8x8 Spearman rank correlation matrix
    across ALL feature specification midpoints.

    CONSERVATIVE STRATEGY:
      - Compute Spearman rho across 49 crop midpoints for all 8 features.
      - Only retain correlations where |rho| >= DEPENDENCY_THRESHOLD (0.25).
      - Set sub-threshold off-diagonal entries to 0.0 (independence assumption).
      - This prevents fabricating correlations not supported by source data.
      - Ensure the resulting matrix is positive-semidefinite.

    NOTE: With only 49 crop spec midpoints, correlation estimates have wide
    confidence intervals. Conservative thresholding is scientifically appropriate.
    """
    print("\n" + "=" * 80)
    print("S4 STEP 4: BUILDING CONSERVATIVE DATA-DRIVEN DEPENDENCY STRUCTURE")
    print("=" * 80)

    # Build matrix of feature midpoints (n_crops x 8)
    X_mids = np.column_stack([feature_mids[feat] for feat in NUMERIC_FEATURES])
    n_crops = X_mids.shape[0]

    # Compute full Spearman correlation matrix
    corr_raw = np.eye(D)
    corr_pvals = np.zeros((D, D))
    dep_analysis_rows = []

    for i in range(D):
        for j in range(i + 1, D):
            rho, pval = spearmanr(X_mids[:, i], X_mids[:, j])
            rho = float(rho) if not np.isnan(rho) else 0.0
            pval = float(pval) if not np.isnan(pval) else 1.0
            corr_raw[i, j] = rho
            corr_raw[j, i] = rho
            corr_pvals[i, j] = pval
            corr_pvals[j, i] = pval

    # Apply conservative thresholding
    corr_conservative = np.eye(D)
    dependency_rows = []

    for i in range(D):
        for j in range(i + 1, D):
            rho = corr_raw[i, j]
            pval = corr_pvals[i, j]
            retained = abs(rho) >= DEPENDENCY_THRESHOLD

            if retained:
                corr_conservative[i, j] = rho
                corr_conservative[j, i] = rho
            else:
                corr_conservative[i, j] = 0.0
                corr_conservative[j, i] = 0.0

            dependency_rows.append({
                'Feature_A':         NUMERIC_FEATURES[i],
                'Feature_B':         NUMERIC_FEATURES[j],
                'Spearman_Rho':      round(rho, 4),
                'P_Value':           round(pval, 4),
                'Abs_Rho':           round(abs(rho), 4),
                'Retained':          retained,
                'Threshold_Used':    DEPENDENCY_THRESHOLD,
                'N_Crops':           n_crops,
                'Source_Note':       'Estimated from 49 crop specification midpoints (range data, not observational)'
            })

    df_dep = pd.DataFrame(dependency_rows)

    # Report retained dependencies
    retained_pairs = df_dep[df_dep['Retained'] == True]
    print(f"  Dependency threshold: |rho| >= {DEPENDENCY_THRESHOLD}")
    print(f"  Total feature pairs: {len(dependency_rows)}")
    print(f"  Retained pairs: {len(retained_pairs)}")
    if len(retained_pairs) > 0:
        print(retained_pairs[['Feature_A', 'Feature_B', 'Spearman_Rho', 'P_Value']].to_string(index=False))
    else:
        print("  No pairs exceeded threshold — using independence (diagonal) structure.")

    # Ensure positive semidefiniteness via eigenvalue clipping
    corr_conservative = _make_psd(corr_conservative)

    print(f"\n[S4] Conservative correlation matrix constructed (PSD guaranteed).")
    return corr_conservative, df_dep, corr_raw


def _make_psd(R, min_eig=1e-6):
    """Ensures a symmetric matrix is positive semidefinite via eigenvalue clipping."""
    R = (R + R.T) / 2.0  # Enforce symmetry
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, min_eig)
    R_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalize to correlation matrix (unit diagonal)
    inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(R_psd), 1e-10)))
    R_corr = inv_sqrt @ R_psd @ inv_sqrt
    np.fill_diagonal(R_corr, 1.0)
    return R_corr


# ==============================================================================
# STEP 5: ADAPTIVE MARGINAL SHAPE PARAMETERS (per crop, per feature)
# ==============================================================================
def get_marginal_params(feat, a, b, feat_analysis_row):
    """
    Derives adaptive marginal distribution parameters for a single feature
    within a single crop's range [a, b].

    Parameters are estimated from:
      1. Range width (b - a)
      2. Midpoint position
      3. Skewness of midpoints across all crops (from analysis row)
      4. Feature type

    Returns: (dist_type, params_dict)
    dist_type options: 'uniform', 'truncnorm', 'beta', 'lhs'
    """
    span    = max(b - a, 1e-8)
    mid     = (a + b) / 2.0
    mid_norm = 0.5  # normalized midpoint is always 0.5 by definition for a crop's own range

    skew_val = float(feat_analysis_row.get('Midpoint_Skewness', 0.0))
    rel_span = float(feat_analysis_row.get('Relative_Span_Ratio', 1.0))

    if feat == 'SOIL_PH':
        # Narrow, continuous — use uniform via Sobol for full coverage
        return 'uniform', {'a': a, 'b': b}

    elif feat == 'CROPDURATION':
        # Wide range — stratified coverage
        return 'lhs', {'a': a, 'b': b}

    elif feat == 'WATERREQUIRED':
        # Bounded, often right-skewed
        if skew_val > 0.4:
            return 'beta', {'a': a, 'b': b, 'alpha': 2.0, 'beta_p': 4.0}
        elif skew_val < -0.4:
            return 'beta', {'a': a, 'b': b, 'alpha': 4.0, 'beta_p': 2.0}
        else:
            return 'beta', {'a': a, 'b': b, 'alpha': 2.5, 'beta_p': 2.5}

    elif feat in ['TEMP', 'RELATIVE_HUMIDITY']:
        # Physiological bell — truncated normal centered at midpoint
        sigma = span / 4.0  # 95% of mass within the range
        return 'truncnorm', {'a': a, 'b': b, 'mu': mid, 'sigma': sigma}

    elif feat in ['N', 'P', 'K']:
        # Nutrient features — handled via copula; use truncnorm as marginal
        sigma = span / 4.0
        return 'truncnorm', {'a': a, 'b': b, 'mu': mid, 'sigma': sigma}

    else:
        # Default: uniform
        return 'uniform', {'a': a, 'b': b}


# ==============================================================================
# STEP 6: GENERATE JOINT CANDIDATES (4 mechanisms in normalized 8D space)
# ==============================================================================
def generate_joint_candidates(ranges, corr_matrix, df_analysis, n_candidates, seed=42):
    """
    Generates a large pool of complete 8D candidate vectors in NORMALIZED [0,1]^8 space.

    Four candidate mechanisms — ALL operating in the joint 8D space:

    A. 8D Scrambled Sobol Sequence
       Low-discrepancy, maximally uniform coverage of [0,1]^8.
       No marginal shaping — pure equidistribution.

    B. 8D Latin Hypercube Sampling
       Stratified in each dimension; no correlation structure imposed.
       Guarantees each marginal is well-covered independently.

    C. Dependency-Aware Gaussian Copula in 8D (data-driven, conservative)
       Uses the conservative correlation matrix from build_dependency_structure().
       Applies adaptive marginal quantile transformations.
       Only non-trivially correlated features receive copula coupling.

    D. Adaptive Hybrid Joint Sampling
       Splits candidates: Sobol base for uniform coverage of 8D space,
       then applies per-feature adaptive marginal reshaping via inversion
       to preserve feature-specific distribution shapes in full 8D.

    All candidates are returned in NORMALIZED [0,1]^8 space.
    """
    n_A = n_candidates // 4
    n_B = n_candidates // 4
    n_C = n_candidates // 4
    n_D = n_candidates - n_A - n_B - n_C

    rng = np.random.RandomState(seed)

    # ---- Mechanism A: 8D Scrambled Sobol ----
    sampler_sobol = qmc.Sobol(d=D, scramble=True, seed=seed)
    U_A = sampler_sobol.random(n=n_A)
    U_A = np.clip(U_A, 1e-6, 1 - 1e-6)

    # ---- Mechanism B: 8D Latin Hypercube ----
    sampler_lhs = qmc.LatinHypercube(d=D, seed=seed + 1)
    U_B = sampler_lhs.random(n=n_B)
    U_B = np.clip(U_B, 1e-6, 1 - 1e-6)

    # ---- Mechanism C: Gaussian Copula (data-driven, conservative) ----
    U_C = _generate_copula_uniform(corr_matrix, n_C, rng)

    # ---- Mechanism D: Adaptive Hybrid Joint ----
    # Start with Sobol base in [0,1]^8, then reshape each marginal
    sampler_d = qmc.Sobol(d=D, scramble=True, seed=seed + 2)
    U_D_base = sampler_d.random(n=n_D)
    U_D_base = np.clip(U_D_base, 1e-6, 1 - 1e-6)
    U_D = _apply_adaptive_marginals(U_D_base, ranges, df_analysis, rng)

    # Stack all candidates in normalized [0,1]^8
    U_all = np.vstack([U_A, U_B, U_C, U_D])

    # Map normalized candidates back to raw feature space for validation
    X_all = denormalize_features(U_all, ranges)

    # Clip to strict bounds (small numerical errors)
    for j, feat in enumerate(NUMERIC_FEATURES):
        a, b = ranges[feat]
        X_all[:, j] = np.clip(X_all[:, j], a, b)

    # Re-normalize after clip (ensuring U_all reflects clipped X)
    U_all = normalize_features(X_all, ranges)

    mechanism_labels = (
        ['A_Sobol'] * n_A +
        ['B_LHS']   * n_B +
        ['C_Copula'] * n_C +
        ['D_Hybrid'] * n_D
    )

    return U_all, X_all, mechanism_labels


def _generate_copula_uniform(corr_matrix, n, rng):
    """
    Generates n samples from a Gaussian copula with the given correlation matrix.
    Returns samples in uniform [0,1]^8 space.
    """
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        corr_matrix = _make_psd(corr_matrix)
        L = np.linalg.cholesky(corr_matrix)

    Z = rng.normal(0, 1, size=(n, D))
    Z_corr = Z @ L.T
    U_corr = norm.cdf(Z_corr)
    return np.clip(U_corr, 1e-6, 1 - 1e-6)


def _apply_adaptive_marginals(U_base, ranges, df_analysis, rng):
    """
    Applies per-feature adaptive marginal inverse CDF transforms to Sobol uniform samples.
    This reshapes the marginals while preserving the joint low-discrepancy structure.

    For each feature:
      u_base (uniform) → inverse CDF of chosen marginal → re-normalize to [0,1]
    """
    U_shaped = np.copy(U_base)

    for j, feat in enumerate(NUMERIC_FEATURES):
        a, b = ranges[feat]
        u = U_base[:, j]

        feat_row = df_analysis[df_analysis['Feature'] == feat].iloc[0].to_dict() if len(
            df_analysis[df_analysis['Feature'] == feat]) > 0 else {}
        dist_type, params = get_marginal_params(feat, a, b, feat_row)

        if dist_type == 'uniform':
            # Already uniform in [0,1] — no reshaping needed
            x_shaped = a + u * (b - a)

        elif dist_type == 'truncnorm':
            mu = params['mu']
            sigma = params['sigma']
            alpha_tn = (a - mu) / sigma if sigma > 0 else -2.0
            beta_tn  = (b - mu) / sigma if sigma > 0 else  2.0
            x_shaped = truncnorm.ppf(u, alpha_tn, beta_tn, loc=mu, scale=sigma)
            x_shaped = np.clip(x_shaped, a, b)

        elif dist_type == 'beta':
            alpha_b = params['alpha']
            beta_b  = params['beta_p']
            x_shaped = a + scipy_beta.ppf(u, alpha_b, beta_b) * (b - a)
            x_shaped = np.clip(x_shaped, a, b)

        elif dist_type == 'lhs':
            # For LHS marginal on Sobol base: apply Beta(2,2) for slight
            # central concentration while preserving stratification spirit
            x_shaped = a + scipy_beta.ppf(u, 2.0, 2.0) * (b - a)
            x_shaped = np.clip(x_shaped, a, b)

        else:
            x_shaped = a + u * (b - a)

        # Re-normalize to [0,1] for this feature
        span = max(b - a, 1e-8)
        U_shaped[:, j] = np.clip((x_shaped - a) / span, 1e-6, 1 - 1e-6)

    return U_shaped


# ==============================================================================
# STEP 7: EVALUATE JOINT CANDIDATES
# ==============================================================================
def evaluate_joint_candidates(U_all, X_all, ranges, corr_matrix):
    """
    Evaluates each complete 8D candidate vector using the S4 joint feasibility score.

    Score components (all computed in normalized space):
      1. boundary_score:       1.0 if strictly within bounds (post-clip, should always be 1)
      2. interior_distance:    Mean distance to nearest boundary in normalized space (0 to 0.5)
      3. marginal_plausibility: Product (geometric mean) of per-feature uniform density proxy
      4. dependency_consistency: How well the joint vector fits the copula structure
      5. pairwise_spread_bonus: Will be added during selection — not computed globally here

    Returns per-candidate scores (n_candidates,) array.
    """
    n = U_all.shape[0]

    # 1. Boundary score (should be ~1.0 for all after clipping)
    boundary_ok = np.ones(n, dtype=float)
    for j, feat in enumerate(NUMERIC_FEATURES):
        a, b = ranges[feat]
        vals = X_all[:, j]
        in_bounds = (vals >= a - BOUNDARY_TOL) & (vals <= b + BOUNDARY_TOL)
        boundary_ok *= in_bounds.astype(float)

    # 2. Interior distance (distance from boundaries in [0,1] space)
    # Distance from lower bound: u, Distance from upper bound: 1-u
    # Interior distance = min(u, 1-u) for each dimension, then mean
    dist_lower = U_all                    # distance from 0
    dist_upper = 1.0 - U_all             # distance from 1
    interior_dist_per_dim = np.minimum(dist_lower, dist_upper)  # (n, D)
    interior_score = np.mean(interior_dist_per_dim, axis=1)     # (n,) in [0, 0.5]
    interior_score = interior_score / 0.5                        # normalize to [0, 1]

    # 3. Dependency consistency
    # For each candidate, compute its Mahalanobis distance from the identity
    # in the copula space (via Gaussian copula: transform u → z = Phi^{-1}(u))
    Z = norm.ppf(U_all)  # (n, D) — standard normal scores
    try:
        L = np.linalg.cholesky(corr_matrix)
        inv_corr = np.linalg.inv(corr_matrix)
    except np.linalg.LinAlgError:
        inv_corr = np.eye(D)

    # Mahalanobis distance squared in the correlated Gaussian space
    md_sq = np.sum((Z @ inv_corr) * Z, axis=1)
    # Convert: lower Mahalanobis distance = better dependency consistency
    # Use chi2 approximation: expected md^2 for d=8 is 8.0
    # Score = exp(-md^2 / (2*D)) — Gaussian density proxy
    dep_consistency = np.exp(-md_sq / (2.0 * D))  # (n,) in [0, 1]
    # Normalize so that the expected value (md^2 = D = 8) gives score 0.607
    # This is already normalized by design of the Gaussian kernel

    # 4. Marginal plausibility
    # For each feature: how plausible is this value within [0,1]?
    # For uniform marginals: all values are equally plausible.
    # For non-uniform (copula-shaped) marginals: penalize extreme values.
    # Approximate by: 1 - |u - 0.5|  (higher near center, lower at extremes)
    # This is a conservative proxy — does NOT invent specific distributions.
    marginal_plaus_per_dim = 1.0 - np.abs(U_all - 0.5) * 2.0  # (n, D) in [0,1]
    # Weight: SOIL_PH needs uniform coverage, so penalize center-bias less
    # NPK: moderate center preference; TEMP/RH: center preference (physiological)
    marginal_weights = np.array([
        0.5,   # SOIL_PH: uniform → low center preference weight
        1.0,   # TEMP
        0.7,   # CROPDURATION
        0.8,   # WATERREQUIRED
        1.0,   # RELATIVE_HUMIDITY
        0.9,   # N
        0.9,   # P
        0.9,   # K
    ])
    marginal_weights = marginal_weights / marginal_weights.sum() * D  # renormalize
    weighted_plaus = marginal_plaus_per_dim * marginal_weights[np.newaxis, :]
    marginal_score = np.mean(weighted_plaus, axis=1)  # (n,) in [0,1]

    # Combined feasibility score (pre-selection, without diversity component)
    # Weights reflect importance: boundary validity must be satisfied; interior
    # distance and dependency consistency are secondary quality metrics.
    base_score = (
        0.40 * boundary_ok +
        0.25 * interior_score +
        0.20 * dep_consistency +
        0.15 * marginal_score
    )

    return base_score, boundary_ok, interior_score, dep_consistency, marginal_score


# ==============================================================================
# STEP 8: CALCULATE JOINT FEASIBILITY AND SELECT DIVERSE SAMPLES
# ==============================================================================
def calculate_joint_feasibility(U_all, base_scores, boundary_ok):
    """
    Finalizes the joint feasibility vector by hard-rejecting boundary violators.
    Candidates with boundary_ok == 0 receive score = -inf (cannot be selected).
    """
    feasibility = np.copy(base_scores)
    feasibility[boundary_ok < 0.5] = -np.inf
    return feasibility


def select_diverse_samples(U_all, X_all, feasibility_scores, n_target, seed=42):
    """
    Greedy Maximin Diversity Selection in normalized 8D space.

    Algorithm:
      1. Filter out infeasible candidates (score = -inf).
      2. Initialize selection with the candidate closest to the centroid
         of feasible candidates (avoids boundary-hugging start).
      3. At each iteration: for each remaining candidate, compute
         min_distance_to_any_selected_point × feasibility_score.
         Select the candidate maximizing this joint criterion.
      4. This simultaneously maximizes spatial coverage (maximin) AND
         feasibility quality — novel joint criterion vs. pure greedy maximin.

    This is the key selection innovation over all three baselines:
      - Baseline-1: selects first N valid rows (no diversity optimization)
      - Baseline-2: no selection at all
      - Baseline-3: jitters duplicates but no maximin selection

    Returns indices of selected candidates.
    """
    rng = np.random.RandomState(seed)
    feasible_mask = feasibility_scores > -np.inf
    feasible_idx = np.where(feasible_mask)[0]

    if len(feasible_idx) < n_target:
        print(f"  [WARNING] Only {len(feasible_idx)} feasible candidates for {n_target} required.")
        return feasible_idx  # Return all feasible (will be handled in fallback)

    U_feasible = U_all[feasible_idx]           # (n_feasible, D)
    scores_f   = feasibility_scores[feasible_idx]  # (n_feasible,)

    # Step 1: Initialize with candidate nearest to centroid of feasible cloud
    centroid = np.mean(U_feasible, axis=0)
    dist_to_centroid = np.linalg.norm(U_feasible - centroid, axis=1)
    first_idx_local = np.argmin(dist_to_centroid)

    selected_local = [first_idx_local]
    min_dist_to_selected = np.full(len(feasible_idx), np.inf)

    # Update min distances from first selected point
    d_first = np.linalg.norm(U_feasible - U_feasible[first_idx_local], axis=1)
    min_dist_to_selected = np.minimum(min_dist_to_selected, d_first)

    # Step 2: Greedy maximin with feasibility weighting
    for k in range(1, n_target):
        # Joint criterion: maximin distance * feasibility score
        # This avoids selecting low-quality candidates just because they are far away
        joint_criterion = min_dist_to_selected * scores_f

        # Exclude already-selected points
        joint_criterion[selected_local] = -np.inf

        next_idx_local = int(np.argmax(joint_criterion))
        selected_local.append(next_idx_local)

        # Update minimum distances
        d_new = np.linalg.norm(U_feasible - U_feasible[next_idx_local], axis=1)
        min_dist_to_selected = np.minimum(min_dist_to_selected, d_new)

        if (k + 1) % 100 == 0:
            pass  # Silent — progress monitored per crop externally

    selected_global = feasible_idx[np.array(selected_local)]
    return selected_global


# ==============================================================================
# STEP 9: VALIDATE SAMPLES
# ==============================================================================
def validate_samples(X_sel, U_sel, ranges, seed=42):
    """
    Performs post-selection validation:
      1. Hard boundary check — all values must be within [a, b].
      2. Exact duplicate detection — remove and replace.
      3. Near-duplicate detection — replace if distance < NEAR_DUPLICATE_THRESHOLD.
      4. Jitter (if needed): small, bounded, applied before rounding.

    Returns validated (X_final, U_final).
    """
    rng = np.random.RandomState(seed)
    X_out = np.copy(X_sel)
    U_out = np.copy(U_sel)
    n = X_out.shape[0]

    # 1. Hard boundary clip
    for j, feat in enumerate(NUMERIC_FEATURES):
        a, b = ranges[feat]
        X_out[:, j] = np.clip(X_out[:, j], a, b)

    # 2. Exact duplicate detection and jitter
    df_check = pd.DataFrame(X_out, columns=NUMERIC_FEATURES)
    dup_mask = df_check.duplicated(keep='first')
    n_dups = dup_mask.sum()

    if n_dups > 0:
        dup_indices = np.where(dup_mask)[0]
        for idx in dup_indices:
            for j, feat in enumerate(NUMERIC_FEATURES):
                a, b = ranges[feat]
                span = max(b - a, 1e-8)
                jitter = rng.uniform(-JITTER_SCALE * span, JITTER_SCALE * span)
                X_out[idx, j] = np.clip(X_out[idx, j] + jitter, a, b)

    # 3. Near-duplicate detection in normalized space
    for j, feat in enumerate(NUMERIC_FEATURES):
        a, b = ranges[feat]
        span = max(b - a, 1e-8)
        U_out[:, j] = (X_out[:, j] - a) / span

    # Detect near-duplicates (pairwise — only for small n)
    # For efficiency, use random subsampling approach:
    # We identify pairs within NEAR_DUPLICATE_THRESHOLD by checking
    # each point against a sample of neighbors only.
    near_dup_count = 0
    if n <= 2000:
        dists = cdist(U_out, U_out, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        too_close = np.where(dists < NEAR_DUPLICATE_THRESHOLD)
        if len(too_close[0]) > 0:
            to_jitter = set(too_close[1])  # jitter the second of each pair
            for idx in to_jitter:
                for j, feat in enumerate(NUMERIC_FEATURES):
                    a, b = ranges[feat]
                    span = max(b - a, 1e-8)
                    jitter = rng.uniform(-JITTER_SCALE * span * 2, JITTER_SCALE * span * 2)
                    X_out[idx, j] = np.clip(X_out[idx, j] + jitter, a, b)
                    U_out[idx, j] = (X_out[idx, j] - a) / span
                near_dup_count += 1

    return X_out, U_out, n_dups, near_dup_count


# ==============================================================================
# STEP 10: MAIN PER-CROP GENERATION PIPELINE
# ==============================================================================
def generate_crop_s4_samples(row, df_analysis, corr_matrix, n_target=N_SAMPLES_PER_CROP, seed=42):
    """
    Full S4 pipeline for a single crop:
      1. Extract ranges
      2. Normalize
      3. Generate joint 8D candidate pool (3× oversampling)
      4. Evaluate joint candidates
      5. Calculate joint feasibility
      6. Select diverse samples via greedy maximin
      7. Validate (bounds, duplicates, near-duplicates)
      8. Fallback if insufficient candidates
    """
    crop_name = str(row['CROPS']).strip()
    ranges = extract_crop_ranges(row)
    n_candidates = OVERSAMPLING_FACTOR * n_target  # default 1800

    # Generate candidate pool in joint 8D normalized space
    U_all, X_all, mech_labels = generate_joint_candidates(
        ranges, corr_matrix, df_analysis, n_candidates, seed=seed
    )

    # Evaluate all candidates as complete 8D vectors
    base_scores, boundary_ok, interior_score, dep_consistency, marginal_score = \
        evaluate_joint_candidates(U_all, X_all, ranges, corr_matrix)

    # Calculate final feasibility
    feasibility = calculate_joint_feasibility(U_all, base_scores, boundary_ok)

    # Select exactly n_target diverse samples
    selected_idx = select_diverse_samples(U_all, X_all, feasibility, n_target, seed=seed)

    # Handle insufficient candidates via fallback
    if len(selected_idx) < n_target:
        needed = n_target - len(selected_idx)
        print(f"  [FALLBACK] {crop_name}: only {len(selected_idx)} feasible; adding {needed} via jitter")
        # Repeat best feasible candidates with small jitter
        best_idx = selected_idx[:min(len(selected_idx), needed)]
        rng_fb = np.random.RandomState(seed + 999)
        X_extra = X_all[best_idx].copy()
        for j, feat in enumerate(NUMERIC_FEATURES):
            a, b = ranges[feat]
            span = max(b - a, 1e-8)
            jitter = rng_fb.uniform(-JITTER_SCALE * span, JITTER_SCALE * span, size=len(X_extra))
            X_extra[:, j] = np.clip(X_extra[:, j] + jitter, a, b)
        X_sel = np.vstack([X_all[selected_idx], X_extra])[:n_target]
        U_sel = normalize_features(X_sel, ranges)
    else:
        X_sel = X_all[selected_idx[:n_target]]
        U_sel = U_all[selected_idx[:n_target]]

    # Validate: bounds, duplicates, near-duplicates
    X_final, U_final, n_dups, n_near_dups = validate_samples(X_sel, U_sel, ranges, seed=seed)

    # Mechanism breakdown for this crop's selected samples
    if len(selected_idx) >= n_target:
        mech_sel = [mech_labels[i] for i in selected_idx[:n_target]]
    else:
        mech_sel = [mech_labels[i] for i in selected_idx] + ['D_Hybrid'] * (n_target - len(selected_idx))

    mech_counts = {
        'A_Sobol': mech_sel.count('A_Sobol'),
        'B_LHS':   mech_sel.count('B_LHS'),
        'C_Copula':mech_sel.count('C_Copula'),
        'D_Hybrid':mech_sel.count('D_Hybrid'),
    }

    return X_final, U_final, n_dups, n_near_dups, mech_counts, ranges


# ==============================================================================
# STEP 11: FULL DATASET GENERATION
# ==============================================================================
def generate_s4_dataset(df_real, df_analysis, corr_matrix):
    """
    Runs the complete S4 pipeline across all crops and assembles final DataFrame.
    """
    print("\n" + "=" * 80)
    print("S4 STEP 11: GENERATING JOINT CONSTRAINT-AWARE SYNTHETIC DATASET")
    print("=" * 80)

    crop_list = df_real['CROPS'].unique()
    n_crops = len(crop_list)
    print(f"  Processing {n_crops} crops × {N_SAMPLES_PER_CROP} samples = {n_crops * N_SAMPLES_PER_CROP} total rows")
    print(f"  Candidate pool: {OVERSAMPLING_FACTOR}× = {OVERSAMPLING_FACTOR * N_SAMPLES_PER_CROP} per crop")
    print(f"  Selection: Greedy Maximin in normalized 8D space\n")

    all_dfs = []
    summary_rows = []
    t_start = time.time()

    for crop_idx, crop in enumerate(crop_list):
        df_c = df_real[df_real['CROPS'] == crop]
        if df_c.empty:
            continue
        row = df_c.iloc[0]

        crop_seed = RANDOM_SEED + crop_idx * 37  # deterministic per crop

        X_final, U_final, n_dups, n_near_dups, mech_counts, ranges = \
            generate_crop_s4_samples(
                row, df_analysis, corr_matrix,
                n_target=N_SAMPLES_PER_CROP, seed=crop_seed
            )

        # === ROUNDING (only after full selection and validation) ===
        # As specified: perform all generation in continuous space, round at the end
        X_rounded = X_final.copy()
        # SOIL_PH: 2 decimal places
        X_rounded[:, 0] = np.round(X_final[:, 0], 2)
        # All other features: round to nearest integer
        for j in range(1, D):
            X_rounded[:, j] = np.round(X_final[:, j]).astype(float)

        # Post-rounding boundary check
        for j, feat in enumerate(NUMERIC_FEATURES):
            a, b = ranges[feat]
            X_rounded[:, j] = np.clip(X_rounded[:, j], a, b)

        # Build DataFrame
        crop_df = pd.DataFrame(X_rounded, columns=NUMERIC_FEATURES)
        crop_df['CROPS'] = crop

        # Add categorical metadata
        for col in CATEGORICAL_FEATURES:
            if col in row.index:
                crop_df[col] = row[col]

        # Post-rounding duplicate check
        n_dups_post_round = crop_df.duplicated(subset=NUMERIC_FEATURES).sum()
        if n_dups_post_round > 0:
            dup_indices = crop_df[crop_df.duplicated(subset=NUMERIC_FEATURES, keep='first')].index
            rng_pr = np.random.RandomState(crop_seed + 500)
            for idx in dup_indices:
                # Jitter SOIL_PH by 0.01 and integer features by ±1 (within bounds)
                a0, b0 = ranges['SOIL_PH']
                new_ph = crop_df.loc[idx, 'SOIL_PH'] + rng_pr.choice([-0.01, 0.01])
                crop_df.loc[idx, 'SOIL_PH'] = round(np.clip(new_ph, a0, b0), 2)
                # Jitter one random integer feature
                rand_feat = rng_pr.choice(['TEMP', 'N', 'P', 'K'])
                j_rf = NUMERIC_FEATURES.index(rand_feat)
                a_r, b_r = ranges[rand_feat]
                new_val = X_rounded[idx - crop_df.index[0], j_rf] + rng_pr.choice([-1, 1])
                crop_df.loc[idx, rand_feat] = int(np.clip(new_val, a_r, b_r))

        all_dfs.append(crop_df)

        # Summary for this crop
        summary_rows.append({
            'Crop':           crop,
            'Sample_Count':   len(crop_df),
            'N_Dups_Pre':     n_dups,
            'N_NearDups_Pre': n_near_dups,
            'N_Dups_Post_Round': n_dups_post_round,
            'Mech_A_Sobol':   mech_counts['A_Sobol'],
            'Mech_B_LHS':     mech_counts['B_LHS'],
            'Mech_C_Copula':  mech_counts['C_Copula'],
            'Mech_D_Hybrid':  mech_counts['D_Hybrid'],
            'SOIL_PH_Mean':   round(float(np.mean(X_rounded[:, 0])), 3),
            'TEMP_Mean':      round(float(np.mean(X_rounded[:, 1])), 1),
            'DURATION_Mean':  round(float(np.mean(X_rounded[:, 2])), 1),
            'WATER_Mean':     round(float(np.mean(X_rounded[:, 3])), 1),
            'RH_Mean':        round(float(np.mean(X_rounded[:, 4])), 1),
            'N_Mean':         round(float(np.mean(X_rounded[:, 5])), 1),
            'P_Mean':         round(float(np.mean(X_rounded[:, 6])), 1),
            'K_Mean':         round(float(np.mean(X_rounded[:, 7])), 1),
        })

        elapsed = time.time() - t_start
        print(f"  [{crop_idx+1:2d}/{n_crops}] {crop:<30s} | {len(crop_df)} samples | "
              f"dups_pre={n_dups} near_dups={n_near_dups} | {elapsed:.1f}s elapsed")

    # Assemble final DataFrame
    df_final = pd.concat(all_dfs, ignore_index=True)

    # Reorder columns
    ordered_cols = ['CROPS'] + CATEGORICAL_FEATURES + NUMERIC_FEATURES
    df_final = df_final[[c for c in ordered_cols if c in df_final.columns]]

    df_summary = pd.DataFrame(summary_rows)

    total_time = time.time() - t_start
    print(f"\n[S4] Generation complete: {len(df_final)} total rows in {total_time:.1f}s")
    return df_final, df_summary


# ==============================================================================
# STEP 12: ANALYZE QUALITY (12 metrics, NO ML)
# ==============================================================================
def analyze_quality(df_synth, df_real, corr_matrix_real, label="S4"):
    """
    Evaluates S4 synthetic dataset using 12 quality metrics (NO ML training).

    Metrics:
    A. Boundary Validity          - % out-of-bounds samples
    B. Range Coverage             - % of each crop's range covered in each dimension
    C. Marginal Distribution Q    - Average Wasserstein distance to spec midpoints
    D. Joint Distribution Q       - Multivariate distributional quality proxy
    E. Dependency Preservation    - Frobenius norm of correlation matrix difference
    F. Normalized Joint Coverage  - Sobol discrepancy proxy (dispersion in 8D)
    G. Maximin Diversity          - Minimum pairwise distance in normalized 8D space
    H. Exact Duplicate Rate       - % exact duplicates
    I. Near-Duplicate Rate        - % near-duplicates (dist < NEAR_DUPLICATE_THRESHOLD)
    J. Outlier Rate               - % samples with Mahalanobis MD^2 > chi2_crit(8, 0.01)=20.09
    K. Cross-Crop Overlap         - Average pairwise range overlap across crops
    L. Crop-wise Feature Coverage - % of each crop's feature range covered per crop
    """
    print("\n" + "=" * 80)
    print("S4 STEP 12: QUALITY ANALYSIS (12 METRICS, NO ML)")
    print("=" * 80)

    quality_results = {}

    total_samples = len(df_synth)
    n_crops = df_synth['CROPS'].nunique()

    # ---- A. Boundary Validity ----
    oob_count = 0
    neg_count = 0
    for crop in df_real['CROPS'].unique():
        df_spec = df_real[df_real['CROPS'] == crop]
        df_c = df_synth[df_synth['CROPS'] == crop]
        if df_spec.empty or df_c.empty:
            continue
        ranges = extract_crop_ranges(df_spec.iloc[0])
        for j, feat in enumerate(NUMERIC_FEATURES):
            a, b = ranges[feat]
            vals = df_c[feat].values.astype(float)
            oob_count += int(np.sum((vals < a - BOUNDARY_TOL) | (vals > b + BOUNDARY_TOL)))
            neg_count += int(np.sum(vals < 0))

    oob_rate = round(oob_count / max(total_samples * D, 1) * 100, 4)
    quality_results['A_OOB_Rate_%'] = oob_rate
    quality_results['A_Negative_Count'] = neg_count
    print(f"  A. Boundary Validity: OOB={oob_rate:.4f}%, Negative={neg_count}")

    # ---- B. Range Coverage ----
    coverage_scores = []
    for crop in df_real['CROPS'].unique():
        df_spec = df_real[df_real['CROPS'] == crop]
        df_c = df_synth[df_synth['CROPS'] == crop]
        if df_spec.empty or df_c.empty:
            continue
        ranges = extract_crop_ranges(df_spec.iloc[0])
        for j, feat in enumerate(NUMERIC_FEATURES):
            a, b = ranges[feat]
            span = max(b - a, 1e-8)
            vals = df_c[feat].values.astype(float)
            actual_min = float(np.min(vals))
            actual_max = float(np.max(vals))
            cov = (actual_max - actual_min) / span
            coverage_scores.append(min(cov, 1.0))

    avg_coverage = round(float(np.mean(coverage_scores)), 4)
    quality_results['B_Avg_Range_Coverage'] = avg_coverage
    print(f"  B. Range Coverage: {avg_coverage:.4f} (1.0=full range covered)")

    # ---- C. Marginal Distribution Quality (Wasserstein) ----
    wasserstein_vals = []
    for feat in NUMERIC_FEATURES:
        min_col, max_col = FEATURE_MAPPINGS[feat]
        real_mids = (df_real[min_col].astype(float) + df_real[max_col].astype(float)) / 2.0
        synth_vals = df_synth[feat].values.astype(float)
        w = wasserstein_distance(real_mids, synth_vals)
        wasserstein_vals.append(w)

    avg_wasserstein = round(float(np.mean(wasserstein_vals)), 4)
    quality_results['C_Avg_Wasserstein'] = avg_wasserstein
    print(f"  C. Avg Wasserstein Distance: {avg_wasserstein:.4f}")

    # ---- D. Joint Distribution Quality ----
    # Proxy: within-crop normalized dispersion (higher = better fill of joint space)
    joint_disp_scores = []
    for crop, group in df_synth.groupby('CROPS'):
        df_spec = df_real[df_real['CROPS'] == crop]
        if df_spec.empty:
            continue
        ranges_c = extract_crop_ranges(df_spec.iloc[0])
        X = group[NUMERIC_FEATURES].values.astype(float)
        X_norm = normalize_features(X, ranges_c)
        # Dispersion = trace of normalized covariance matrix (higher = better joint fill)
        if len(X_norm) > 1:
            cov_n = np.cov(X_norm, rowvar=False)
            disp = float(np.trace(cov_n))
        else:
            disp = 0.0
        joint_disp_scores.append(disp)

    avg_joint_disp = round(float(np.mean(joint_disp_scores)), 4)
    quality_results['D_Avg_Joint_Dispersion'] = avg_joint_disp
    print(f"  D. Avg Joint Dispersion (trace of norm cov): {avg_joint_disp:.4f}")

    # ---- E. Dependency Preservation ----
    synth_corr = df_synth[NUMERIC_FEATURES].corr(method='spearman').values
    frob_error = round(float(np.linalg.norm(corr_matrix_real - synth_corr, ord='fro')), 4)
    quality_results['E_Corr_Frobenius_Error'] = frob_error
    print(f"  E. Correlation Frobenius Error: {frob_error:.4f}")

    # ---- F. Normalized Joint Coverage (Sobol Discrepancy proxy) ----
    # Use star discrepancy approximation via max deviation from uniform
    coverage_vals = []
    for crop, group in df_synth.groupby('CROPS'):
        df_spec = df_real[df_real['CROPS'] == crop]
        if df_spec.empty or len(group) < 10:
            continue
        ranges_c = extract_crop_ranges(df_spec.iloc[0])
        X_n = normalize_features(group[NUMERIC_FEATURES].values.astype(float), ranges_c)
        # Compute fill ratio: fraction of 8D [0,1] hypercube "covered"
        # Approximate by checking n_bins^(1/D) subdivisions
        n_bins = 4  # 4^(1/8) ≈ 1.19 bins per dimension → feasible
        bin_edges = np.linspace(0, 1, n_bins + 1)
        total_cells = n_bins ** D
        occupied = set()
        for x in X_n:
            cell = tuple(int(min(np.searchsorted(bin_edges[1:], xi), n_bins - 1)) for xi in x)
            occupied.add(cell)
        fill = len(occupied) / total_cells
        coverage_vals.append(fill)

    avg_fill = round(float(np.mean(coverage_vals)) if coverage_vals else 0.0, 4)
    quality_results['F_Norm_Joint_Coverage'] = avg_fill
    print(f"  F. Normalized 8D Joint Coverage: {avg_fill:.4f} (1.0=all {4**D} cells filled)")

    # ---- G. Maximin Diversity ----
    maximin_dists = []
    for crop, group in df_synth.groupby('CROPS'):
        df_spec = df_real[df_real['CROPS'] == crop]
        if df_spec.empty or len(group) < 2:
            continue
        ranges_c = extract_crop_ranges(df_spec.iloc[0])
        X_n = normalize_features(group[NUMERIC_FEATURES].values.astype(float), ranges_c)
        if len(X_n) > 500:
            idx_sub = np.random.choice(len(X_n), 500, replace=False)
            X_n = X_n[idx_sub]
        dists = cdist(X_n, X_n, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        maximin_dists.append(float(np.min(min_dists)))

    avg_maximin = round(float(np.mean(maximin_dists)) if maximin_dists else 0.0, 6)
    quality_results['G_Avg_Maximin_Distance'] = avg_maximin
    print(f"  G. Avg Maximin Diversity (8D normalized): {avg_maximin:.6f}")

    # ---- H. Exact Duplicate Rate ----
    dup_count = int(df_synth.duplicated(subset=NUMERIC_FEATURES).sum())
    dup_rate = round(dup_count / max(total_samples, 1) * 100, 4)
    quality_results['H_Exact_Dup_Rate_%'] = dup_rate
    print(f"  H. Exact Duplicate Rate: {dup_rate:.4f}%")

    # ---- I. Near-Duplicate Rate ----
    near_dup_count = 0
    for crop, group in df_synth.groupby('CROPS'):
        df_spec = df_real[df_real['CROPS'] == crop]
        if df_spec.empty or len(group) < 2:
            continue
        ranges_c = extract_crop_ranges(df_spec.iloc[0])
        X_n = normalize_features(group[NUMERIC_FEATURES].values.astype(float), ranges_c)
        if len(X_n) > 300:
            idx_sub = np.random.choice(len(X_n), 300, replace=False)
            X_n = X_n[idx_sub]
        dists = cdist(X_n, X_n, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        near_dup_count += int(np.sum(np.min(dists, axis=1) < NEAR_DUPLICATE_THRESHOLD))

    near_dup_rate = round(near_dup_count / max(total_samples, 1) * 100, 4)
    quality_results['I_Near_Dup_Rate_%'] = near_dup_rate
    print(f"  I. Near-Duplicate Rate: {near_dup_rate:.4f}%")

    # ---- J. Outlier Rate (Mahalanobis MD^2 > 20.09) ----
    outlier_count = 0
    for crop, group in df_synth.groupby('CROPS'):
        X = group[NUMERIC_FEATURES].values.astype(float)
        if len(X) < 10:
            continue
        mean_v = np.mean(X, axis=0)
        cov_m  = np.cov(X, rowvar=False) + np.eye(D) * 1e-6
        inv_cov= np.linalg.pinv(cov_m)
        diff   = X - mean_v
        md_sq  = np.sum((diff @ inv_cov) * diff, axis=1)
        outlier_count += int(np.sum(md_sq > 20.09))

    outlier_rate = round(outlier_count / max(total_samples, 1) * 100, 4)
    quality_results['J_Outlier_Rate_%'] = outlier_rate
    print(f"  J. Outlier Rate (MD^2 > 20.09): {outlier_rate:.4f}%")

    # ---- K. Cross-Crop Overlap ----
    crops_list = df_real['CROPS'].unique()
    overlap_vals = []
    for fi, feat in enumerate(NUMERIC_FEATURES):
        min_col, max_col = FEATURE_MAPPINGS[feat]
        for i in range(len(crops_list)):
            row_i = df_real[df_real['CROPS'] == crops_list[i]]
            if row_i.empty:
                continue
            a1 = float(row_i[min_col].iloc[0])
            b1 = float(row_i[max_col].iloc[0])
            for j in range(i + 1, len(crops_list)):
                row_j = df_real[df_real['CROPS'] == crops_list[j]]
                if row_j.empty:
                    continue
                a2 = float(row_j[min_col].iloc[0])
                b2 = float(row_j[max_col].iloc[0])
                inter = max(0.0, min(b1, b2) - max(a1, a2))
                union = max(b1, b2) - min(a1, a2)
                if union > 0:
                    overlap_vals.append(inter / union)

    avg_overlap = round(float(np.mean(overlap_vals)) if overlap_vals else 0.0, 4)
    quality_results['K_Cross_Crop_Overlap'] = avg_overlap
    print(f"  K. Cross-Crop Spec Overlap (avg IoU per feat): {avg_overlap:.4f}")

    # ---- L. Crop-wise Feature-Space Coverage ----
    # Average per-feature range coverage per crop (already computed in B but per-crop here)
    per_crop_coverage = []
    for crop in df_real['CROPS'].unique():
        df_spec = df_real[df_real['CROPS'] == crop]
        df_c = df_synth[df_synth['CROPS'] == crop]
        if df_spec.empty or df_c.empty:
            continue
        ranges_c = extract_crop_ranges(df_spec.iloc[0])
        covs = []
        for feat in NUMERIC_FEATURES:
            a, b = ranges_c[feat]
            span = max(b - a, 1e-8)
            vals = df_c[feat].values.astype(float)
            cov = min((np.max(vals) - np.min(vals)) / span, 1.0)
            covs.append(cov)
        per_crop_coverage.append(float(np.mean(covs)))

    avg_per_crop_cov = round(float(np.mean(per_crop_coverage)) if per_crop_coverage else 0.0, 4)
    quality_results['L_Avg_PerCrop_Coverage'] = avg_per_crop_cov
    print(f"  L. Avg Per-Crop Feature Coverage: {avg_per_crop_cov:.4f}")

    quality_results['Label'] = label
    quality_results['Total_Samples'] = total_samples
    quality_results['N_Crops'] = n_crops

    # Composite Synthetic Quality Score (SQS)
    # Weights: validity (30) + coverage (15) + marginal (10) + joint (10) + dep (10) + 
    #          diversity_maximin (10) + low_dups (10) + low_outliers (5)
    norm_wass = min(avg_wasserstein / 50.0, 1.0)
    norm_corr = min(frob_error / 3.0, 1.0)
    sqs = (
        30.0 * (1.0 - oob_rate / 100.0) +
        15.0 * avg_coverage +
        10.0 * (1.0 - norm_wass) +
        10.0 * (1.0 - norm_corr) +
        10.0 * min(avg_maximin / 0.1, 1.0) +
        10.0 * avg_fill +
        10.0 * (1.0 - dup_rate / 100.0) +
        5.0  * (1.0 - outlier_rate / 100.0)
    )
    quality_results['SQS_Score'] = round(sqs, 2)
    print(f"\n[S4] Synthetic Quality Score (SQS): {sqs:.2f} / 100.0")

    return quality_results


# ==============================================================================
# STEP 13: CANDIDATE STRATEGY COMPARISON
# ==============================================================================
def compare_candidate_strategies(df_real, df_analysis, corr_matrix, sample_crop_names=None):
    """
    Evaluates each of the 4 candidate generation mechanisms (A/B/C/D) independently
    for a sample of crops using quality metrics. Returns comparison DataFrame.

    This uses a multi-objective evaluation — NOT just diversity alone.
    Avoids arbitrary weighting by reporting each metric separately.
    """
    print("\n" + "=" * 80)
    print("S4 STEP 13: CANDIDATE STRATEGY COMPARISON")
    print("=" * 80)

    # Use a subset of crops for efficiency
    all_crops = df_real['CROPS'].unique()
    if sample_crop_names is None:
        n_sample = min(10, len(all_crops))
        sample_crops = all_crops[:n_sample]
    else:
        sample_crops = sample_crop_names

    n_per_mech = N_SAMPLES_PER_CROP  # Compare on full 600 per mechanism

    comp_rows = []

    for crop in sample_crops:
        df_c = df_real[df_real['CROPS'] == crop]
        if df_c.empty:
            continue
        row = df_c.iloc[0]
        ranges = extract_crop_ranges(row)
        n_cand = OVERSAMPLING_FACTOR * N_SAMPLES_PER_CROP

        U_all, X_all, mech_labels = generate_joint_candidates(
            ranges, corr_matrix, df_analysis, n_cand, seed=RANDOM_SEED
        )
        base_scores, boundary_ok, interior_score, dep_consistency, marginal_score = \
            evaluate_joint_candidates(U_all, X_all, ranges, corr_matrix)

        # Evaluate each mechanism separately
        mech_arr = np.array(mech_labels)
        for mech in ['A_Sobol', 'B_LHS', 'C_Copula', 'D_Hybrid']:
            mech_mask = mech_arr == mech
            mech_idx = np.where(mech_mask)[0]
            if len(mech_idx) == 0:
                continue

            U_m = U_all[mech_idx]
            X_m = X_all[mech_idx]

            # Boundary validity
            oob = 0
            for j, feat in enumerate(NUMERIC_FEATURES):
                a, b = ranges[feat]
                oob += int(np.sum((X_m[:, j] < a - BOUNDARY_TOL) | (X_m[:, j] > b + BOUNDARY_TOL)))
            oob_r = round(oob / max(len(mech_idx) * D, 1) * 100, 4)

            # Interior distance (avg)
            id_mean = round(float(np.mean(interior_score[mech_idx])), 4)

            # Dependency consistency (avg)
            dc_mean = round(float(np.mean(dep_consistency[mech_idx])), 4)

            # Marginal plausibility (avg)
            mp_mean = round(float(np.mean(marginal_score[mech_idx])), 4)

            # Diversity: avg nearest-neighbor distance in normalized space
            n_sub = min(200, len(U_m))
            idx_sub = np.random.choice(len(U_m), n_sub, replace=False)
            dists = cdist(U_m[idx_sub], U_m[idx_sub], metric='euclidean')
            np.fill_diagonal(dists, np.inf)
            nn_dist = round(float(np.mean(np.min(dists, axis=1))), 6)

            # Range coverage (average across features)
            covs = []
            for j, feat in enumerate(NUMERIC_FEATURES):
                a, b = ranges[feat]
                span = max(b - a, 1e-8)
                cov = min((np.max(X_m[:, j]) - np.min(X_m[:, j])) / span, 1.0)
                covs.append(cov)
            range_cov = round(float(np.mean(covs)), 4)

            comp_rows.append({
                'Crop':                    crop,
                'Mechanism':               mech,
                'N_Candidates':            len(mech_idx),
                'OOB_Rate_%':             oob_r,
                'Interior_Distance_Avg':   id_mean,
                'Dep_Consistency_Avg':     dc_mean,
                'Marginal_Plaus_Avg':      mp_mean,
                'NN_Distance_Avg':         nn_dist,
                'Range_Coverage_Avg':      range_cov,
            })

    df_comp = pd.DataFrame(comp_rows)

    # Aggregate across crops
    agg = df_comp.groupby('Mechanism').agg({
        'OOB_Rate_%':             'mean',
        'Interior_Distance_Avg':  'mean',
        'Dep_Consistency_Avg':    'mean',
        'Marginal_Plaus_Avg':     'mean',
        'NN_Distance_Avg':        'mean',
        'Range_Coverage_Avg':     'mean',
    }).round(4)

    print("\nCandidate Strategy Comparison (averaged across crops):")
    print(agg.to_string())

    return df_comp, agg


# ==============================================================================
# STEP 14: JOINT SPACE ANALYSIS
# ==============================================================================
def analyze_joint_space(df_synth, df_real):
    """
    Analyzes the joint 8D feature space properties of the S4 synthetic dataset.
    """
    print("\n" + "=" * 80)
    print("S4 STEP 14: JOINT SPACE ANALYSIS")
    print("=" * 80)

    rows = []

    for crop in df_real['CROPS'].unique():
        df_spec = df_real[df_real['CROPS'] == crop]
        df_c = df_synth[df_synth['CROPS'] == crop]
        if df_spec.empty or len(df_c) < 2:
            continue

        ranges_c = extract_crop_ranges(df_spec.iloc[0])
        X_raw = df_c[NUMERIC_FEATURES].values.astype(float)
        X_norm = normalize_features(X_raw, ranges_c)

        # Volume coverage: convex hull approximation via PCA variances
        cov_n = np.cov(X_norm, rowvar=False) + np.eye(D) * 1e-9
        eigvals = np.linalg.eigvalsh(cov_n)
        eigvals = np.maximum(eigvals, 0)
        # Effective dimensionality (participation ratio)
        pr = (np.sum(eigvals) ** 2) / max(np.sum(eigvals ** 2), 1e-12)

        # Nearest-neighbor distances
        n_sub = min(200, len(X_norm))
        idx_sub = np.random.choice(len(X_norm), n_sub, replace=False)
        dists_sub = cdist(X_norm[idx_sub], X_norm[idx_sub], metric='euclidean')
        np.fill_diagonal(dists_sub, np.inf)
        nn_dists = np.min(dists_sub, axis=1)

        # Range fill per feature
        fill_per_feat = []
        for j, feat in enumerate(NUMERIC_FEATURES):
            a, b = ranges_c[feat]
            span = max(b - a, 1e-8)
            vals = X_raw[:, j]
            fill = min((np.max(vals) - np.min(vals)) / span, 1.0)
            fill_per_feat.append(fill)

        rows.append({
            'Crop':                    crop,
            'N_Samples':               len(df_c),
            'Effective_Dimensionality': round(float(pr), 4),
            'Max_Eigenvalue':          round(float(np.max(eigvals)), 4),
            'Min_NN_Distance_8D':      round(float(np.min(nn_dists)), 6),
            'Mean_NN_Distance_8D':     round(float(np.mean(nn_dists)), 6),
            'Max_NN_Distance_8D':      round(float(np.max(nn_dists)), 6),
            **{f'Fill_{feat}': round(fill_per_feat[j], 4)
               for j, feat in enumerate(NUMERIC_FEATURES)},
            'Avg_Fill':                round(float(np.mean(fill_per_feat)), 4),
        })

    df_joint = pd.DataFrame(rows)
    print(df_joint[['Crop', 'Effective_Dimensionality', 'Mean_NN_Distance_8D', 'Avg_Fill']].to_string(index=False))
    return df_joint


# ==============================================================================
# STEP 15: SAVE OUTPUTS
# ==============================================================================
def save_outputs(df_final, df_summary, df_analysis, df_dep, df_dep_raw,
                 df_cand_comp, df_joint, quality_dict):
    """Saves all 7 Excel output files."""
    print("\n" + "=" * 80)
    print("S4 STEP 15: SAVING OUTPUT FILES")
    print("=" * 80)

    # 1. Main synthetic dataset
    df_final.to_excel(OUTPUT_SYNTHETIC_EXCEL, index=False)
    print(f"[OK] {OUTPUT_SYNTHETIC_EXCEL}")

    # 2. Feature analysis
    df_analysis.to_excel(OUTPUT_FEATURE_ANALYSIS, index=False)
    print(f"[OK] {OUTPUT_FEATURE_ANALYSIS}")

    # 3. Dependency analysis
    with pd.ExcelWriter(OUTPUT_DEPENDENCY_ANALYSIS) as writer:
        df_dep.to_excel(writer, sheet_name='Conservative_Pairs', index=False)
        df_raw_corr = pd.DataFrame(df_dep_raw, index=NUMERIC_FEATURES, columns=NUMERIC_FEATURES)
        df_raw_corr.to_excel(writer, sheet_name='Raw_Spearman_Matrix')
    print(f"[OK] {OUTPUT_DEPENDENCY_ANALYSIS}")

    # 4. Candidate comparison
    df_cand_comp.to_excel(OUTPUT_CANDIDATE_COMP, index=False)
    print(f"[OK] {OUTPUT_CANDIDATE_COMP}")

    # 5. Joint space analysis
    df_joint.to_excel(OUTPUT_JOINT_SPACE, index=False)
    print(f"[OK] {OUTPUT_JOINT_SPACE}")

    # 6. Quality analysis
    df_quality = pd.DataFrame([quality_dict])
    df_quality.to_excel(OUTPUT_QUALITY_ANALYSIS, index=False)
    print(f"[OK] {OUTPUT_QUALITY_ANALYSIS}")

    # 7. Generation summary
    df_summary.to_excel(OUTPUT_GEN_SUMMARY, index=False)
    print(f"[OK] {OUTPUT_GEN_SUMMARY}")


# ==============================================================================
# STEP 16: GENERATE PLOTS
# ==============================================================================
def generate_plots(df_real, df_synth, df_joint, df_cand_comp, quality_dict, corr_matrix_real):
    """Generates useful diagnostic plots for S4."""
    print("\n" + "=" * 80)
    print("S4 STEP 16: GENERATING PLOTS")
    print("=" * 80)

    # ---- Plot 1: Feature Distributions (Spec Midpoints vs S4 Synthetic) ----
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, feat in enumerate(NUMERIC_FEATURES):
        min_col, max_col = FEATURE_MAPPINGS[feat]
        real_mids = (df_real[min_col].astype(float) + df_real[max_col].astype(float)) / 2.0
        synth_vals = df_synth[feat].values.astype(float)
        ax = axes[i]
        sns.kdeplot(real_mids,  ax=ax, label='Spec Midpoints', color='navy', linewidth=2.5)
        sns.kdeplot(synth_vals, ax=ax, label='S4 Synthetic',   color='darkorange', linestyle='--', linewidth=2)
        ax.set_title(f"{feat}", fontweight='bold')
        ax.legend(fontsize=7)
        ax.set_xlabel('Value')
    plt.suptitle("S4: Feature Distributions — Spec Midpoints vs Synthetic", fontweight='bold', fontsize=12)
    plt.tight_layout()
    p1 = os.path.join(DATASET_DIR, 'S4_plot_feature_distributions.png')
    plt.savefig(p1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {p1}")

    # ---- Plot 2: Normalized 8D Space Coverage (PCA projection per crop sample) ----
    fig, ax = plt.subplots(figsize=(10, 7))
    crops_sample = df_real['CROPS'].unique()[:12]
    cmap = plt.cm.get_cmap('tab20', len(crops_sample))
    for ci, crop in enumerate(crops_sample):
        df_spec = df_real[df_real['CROPS'] == crop]
        df_c = df_synth[df_synth['CROPS'] == crop]
        if df_spec.empty or len(df_c) < 2:
            continue
        ranges_c = extract_crop_ranges(df_spec.iloc[0])
        X_raw = df_c[NUMERIC_FEATURES].values.astype(float)
        X_norm = normalize_features(X_raw, ranges_c)
        # Simple 2D projection: dim 0 (SOIL_PH normalized) vs dim 5 (N normalized)
        ax.scatter(X_norm[:, 0], X_norm[:, 5], alpha=0.3, s=8, color=cmap(ci), label=crop)
    ax.set_xlabel('SOIL_PH (normalized)')
    ax.set_ylabel('N (normalized)')
    ax.set_title('S4: Normalized Feature Space Coverage (SOIL_PH vs N) — First 12 Crops')
    ax.legend(loc='upper right', fontsize=6, ncol=2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    p2 = os.path.join(DATASET_DIR, 'S4_plot_normalized_coverage.png')
    plt.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {p2}")

    # ---- Plot 3: Correlation / Dependency Matrix ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    df_real_corr = pd.DataFrame(corr_matrix_real, index=NUMERIC_FEATURES, columns=NUMERIC_FEATURES)
    df_synth_corr = df_synth[NUMERIC_FEATURES].corr(method='spearman')
    sns.heatmap(df_real_corr,  ax=axes[0], annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, cbar=False)
    axes[0].set_title("Source Specification Rank Correlation\n(Conservative Thresholded)", fontweight='bold')
    sns.heatmap(df_synth_corr, ax=axes[1], annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, cbar=True)
    axes[1].set_title("S4 Synthetic Dataset Rank Correlation", fontweight='bold')
    plt.suptitle("S4: Dependency Structure Preservation", fontweight='bold')
    plt.tight_layout()
    p3 = os.path.join(DATASET_DIR, 'S4_plot_correlation_matrices.png')
    plt.savefig(p3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {p3}")

    # ---- Plot 4: Per-Crop Feature Space Coverage (Heatmap) ----
    fill_cols = [f'Fill_{feat}' for feat in NUMERIC_FEATURES]
    df_fill = df_joint[['Crop'] + fill_cols].set_index('Crop')
    df_fill.columns = NUMERIC_FEATURES
    fig, ax = plt.subplots(figsize=(14, max(6, len(df_fill) * 0.2)))
    sns.heatmap(df_fill, ax=ax, cmap='YlOrRd', vmin=0, vmax=1,
                annot=False, linewidths=0.3)
    ax.set_title("S4: Per-Crop Feature Range Coverage (1.0 = full range covered)",
                 fontweight='bold')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Crop')
    plt.tight_layout()
    p4 = os.path.join(DATASET_DIR, 'S4_plot_crop_coverage_heatmap.png')
    plt.savefig(p4, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {p4}")

    # ---- Plot 5: Candidate Strategy Comparison ----
    if len(df_cand_comp) > 0:
        metrics_to_plot = ['OOB_Rate_%', 'NN_Distance_Avg', 'Range_Coverage_Avg',
                           'Interior_Distance_Avg', 'Dep_Consistency_Avg']
        agg_comp = df_cand_comp.groupby('Mechanism')[metrics_to_plot].mean().reset_index()

        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 5))
        for ai, metric in enumerate(metrics_to_plot):
            sns.barplot(data=agg_comp, x='Mechanism', y=metric,
                        palette='Set2', ax=axes[ai])
            axes[ai].set_title(metric.replace('_', '\n'), fontsize=9, fontweight='bold')
            axes[ai].set_xlabel('')
            axes[ai].tick_params(axis='x', rotation=20)
        plt.suptitle("S4: Candidate Generation Strategy Comparison (Multi-Objective)",
                     fontweight='bold', fontsize=12)
        plt.tight_layout()
        p5 = os.path.join(DATASET_DIR, 'S4_plot_candidate_comparison.png')
        plt.savefig(p5, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] {p5}")

    # ---- Plot 6: Joint Space Diversity (Maximin distances per crop) ----
    if 'Mean_NN_Distance_8D' in df_joint.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(range(len(df_joint)), df_joint['Min_NN_Distance_8D'],
                    color='steelblue', alpha=0.8)
        axes[0].set_title("Per-Crop Minimum NN Distance (8D normalized)\n(Higher = More Diverse)",
                           fontweight='bold')
        axes[0].set_xlabel('Crop Index')
        axes[0].set_ylabel('Min NN Distance')

        axes[1].bar(range(len(df_joint)), df_joint['Avg_Fill'],
                    color='forestgreen', alpha=0.8)
        axes[1].set_title("Per-Crop Average Feature Range Coverage\n(1.0 = Full Range Covered)",
                           fontweight='bold')
        axes[1].set_xlabel('Crop Index')
        axes[1].set_ylabel('Coverage Fraction')

        plt.suptitle("S4: Joint Space Diversity Metrics", fontweight='bold', fontsize=12)
        plt.tight_layout()
        p6 = os.path.join(DATASET_DIR, 'S4_plot_joint_diversity.png')
        plt.savefig(p6, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] {p6}")

    # ---- Plot 7: Quality Metrics Radar/Bar Summary ----
    metric_labels = ['OOB=0%', 'Coverage', '1-Wass', '1-Frob', 'Maximin', 'Joint Fill', 'No Dups', 'No Outliers']
    oob_rate = quality_dict.get('A_OOB_Rate_%', 0.0)
    avg_cov  = quality_dict.get('B_Avg_Range_Coverage', 0.0)
    norm_wass= min(quality_dict.get('C_Avg_Wasserstein', 0.0) / 50.0, 1.0)
    frob_err = min(quality_dict.get('E_Corr_Frobenius_Error', 0.0) / 3.0, 1.0)
    maximin  = min(quality_dict.get('G_Avg_Maximin_Distance', 0.0) / 0.1, 1.0)
    jfill    = quality_dict.get('F_Norm_Joint_Coverage', 0.0)
    dup_rate = quality_dict.get('H_Exact_Dup_Rate_%', 0.0)
    out_rate = quality_dict.get('J_Outlier_Rate_%', 0.0)

    vals = [
        1.0 - oob_rate / 100.0,
        avg_cov,
        1.0 - norm_wass,
        1.0 - frob_err,
        maximin,
        jfill,
        1.0 - dup_rate / 100.0,
        1.0 - out_rate / 100.0,
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = range(len(metric_labels))
    colors = ['#2ecc71' if v >= 0.8 else '#f39c12' if v >= 0.5 else '#e74c3c' for v in vals]
    bars = ax.bar(x_pos, vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(metric_labels, rotation=25, ha='right')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score (1.0 = best)')
    ax.set_title(f"S4: Quality Metric Summary (SQS = {quality_dict.get('SQS_Score', 0):.2f}/100)",
                 fontweight='bold')
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='0.8 threshold')
    ax.legend()
    plt.tight_layout()
    p7 = os.path.join(DATASET_DIR, 'S4_plot_quality_summary.png')
    plt.savefig(p7, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {p7}")


# ==============================================================================
# STEP 17: GENERATE CONCLUSION REPORT
# ==============================================================================
def generate_report(df_real, df_analysis, df_dep, df_dep_agg_str, quality_dict, df_summary, elapsed_total):
    """
    Generates S4_conclusion_report.md with honest, complete methodology description.
    """
    n_retained = df_dep['Retained'].sum() if 'Retained' in df_dep.columns else 0
    n_total_pairs = len(df_dep) if len(df_dep) > 0 else D * (D - 1) // 2

    report = f"""# S4 — Joint Constraint and Boundary-Aware Synthetic Data Generation
## Final Methodology Report

> **IMPORTANT LIMITATION STATEMENT**
> The source dataset (`crop-dataset.xlsx`) is a **range/specification-based dataset** — it contains
> minimum and maximum bounds for approximately 49 crop classes, NOT thousands of observational measurements.
> S4 does NOT claim to have learned the true real-world distribution.
> S4 does NOT claim to have discovered real correlations unless the source specification supports them.
> All dependency estimates are derived from 49 crop midpoint values (one per crop) — a conservative estimate.
> S4 does NOT train any machine-learning model during generation.

---

## A. Methodology

### A.1 What the Three Baseline Frameworks Do

| Framework | Core Strategy | Dependency | Selection |
|-----------|--------------|-----------|-----------|
| **Baseline-1** (10_Synthetic_Crop_Data_AHAPSF1.py) | Joint 8D Sobol → full 8-feature Gaussian copula with HARD-CODED crop-group correlation matrices | Full 8-feature copula; 5 hard-coded crop groups | First N valid rows after 1.2× oversampling |
| **Baseline-2** (7_Synthetic_Crop_Data_AHAPSF.py) | Feature-by-feature independent: Sobol/LHS/TruncNorm/Beta/Copula | NPK only; HARD-CODED corr=0.5 | Direct 600 — no selection, no deduplication |
| **Baseline-3** (AHAPSF2/3_framework.py) | Adaptive per-feature sampler selection (data-driven algorithm choice) | NPK Spearman copula; other features independent | Jitter on duplicates only; no maximin selection |

### A.2 Main Limitations of the Baseline Frameworks

1. **Feature-wise concatenation as main strategy**: All three generate features independently
   (or at most group NPK via copula) and concatenate. No framework selects final samples
   based on joint 8D vector quality.

2. **No 8D maximin diversity selection**: Baseline-1 selects the first N valid rows.
   Baseline-2 generates exactly N with no selection. Baseline-3 jitters duplicates but
   does not optimize joint-space diversity.

3. **Fabricated or hardcoded correlations**: Baseline-1 uses hard-coded crop-group correlation
   matrices (cereals/pulses/oilseeds/vegetables/commercial) with fixed values not derived from data.
   Baseline-2 hard-codes corr=0.5 for NPK for all crops regardless of actual structure.

4. **Fixed distribution parameters**: Baseline-2 uses Beta(5,5) for all crops regardless of
   skewness. Baseline-1 uses fixed copula regardless of feature independence.

### A.3 What S4 Changes

S4 operates on COMPLETE 8-dimensional feature vectors, not individual features:

1. **Joint 8D candidate generation**: Four mechanisms — Sobol-8D, LHS-8D, Gaussian Copula-8D,
   and Adaptive Hybrid-8D — all generate complete 8D vectors (not individual features).

2. **Conservative data-driven dependency**: Spearman correlations are computed across all
   49 crop midpoints. Only correlations with |rho| >= {DEPENDENCY_THRESHOLD} are preserved.
   This prevents fabricating correlations not supported by source data.

3. **Joint feasibility score**: Each complete 8D candidate vector is scored holistically:
   boundary validity + interior distance + dependency consistency + marginal plausibility.

4. **Greedy maximin diversity selection**: 600 samples are selected from 1,800 candidates
   by maximizing minimum pairwise distance in normalized 8D space, weighted by feasibility.
   This is fundamentally different from all three baselines.

5. **Range-aware normalization**: All joint-space operations use normalized [0,1]^8 space,
   preventing large-scale features (WATERREQUIRED, N) from dominating distance calculations.

### A.4 Why Joint Feature-Space Generation May Help Downstream Classification

When features are generated independently and concatenated, no mechanism prevents unlikely
feature combinations (e.g., very high temperature with very high relative humidity AND very
low water requirement simultaneously). Joint generation with feasibility scoring reduces
such combinations.

The greedy maximin selection ensures maximum spread across the 8D feature space per crop,
which provides denser, more informative training signal for downstream ML classifiers
operating on all 8 features simultaneously.

---

## B. Novelty

The novelty of S4 is NOT merely combining existing algorithms. The novel contribution is:

1. **Complete 8D vector evaluation**: The joint feasibility score treats each candidate as
   a whole 8D entity, not independent scalar values.

2. **Maximin × Feasibility joint selection criterion**: No baseline uses this. The selection
   criterion simultaneously maximizes spatial diversity AND respects distribution quality.

3. **Conservative thresholded dependency**: Instead of hard-coding correlations (BL-1) or
   assuming independence (BL-2, BL-3 for non-NPK), S4 uses a principled threshold on
   data-derived Spearman correlations.

4. **4-mechanism joint candidate pool**: Operating all 4 mechanisms in full 8D normalized
   space (vs. feature-by-feature generation) creates a richer, more diverse candidate pool
   from which the best 600 can be selected.

---

## C. Quality Results

| Metric | Value |
|--------|-------|
| A. Boundary Validity (OOB Rate) | {quality_dict.get('A_OOB_Rate_%', 'N/A')}% |
| A. Negative Values | {quality_dict.get('A_Negative_Count', 'N/A')} |
| B. Avg Range Coverage | {quality_dict.get('B_Avg_Range_Coverage', 'N/A')} |
| C. Avg Wasserstein Distance | {quality_dict.get('C_Avg_Wasserstein', 'N/A')} |
| D. Avg Joint Dispersion | {quality_dict.get('D_Avg_Joint_Dispersion', 'N/A')} |
| E. Correlation Frobenius Error | {quality_dict.get('E_Corr_Frobenius_Error', 'N/A')} |
| F. Normalized 8D Joint Coverage | {quality_dict.get('F_Norm_Joint_Coverage', 'N/A')} |
| G. Avg Maximin Distance | {quality_dict.get('G_Avg_Maximin_Distance', 'N/A')} |
| H. Exact Duplicate Rate | {quality_dict.get('H_Exact_Dup_Rate_%', 'N/A')}% |
| I. Near-Duplicate Rate | {quality_dict.get('I_Near_Dup_Rate_%', 'N/A')}% |
| J. Outlier Rate | {quality_dict.get('J_Outlier_Rate_%', 'N/A')}% |
| K. Cross-Crop Overlap | {quality_dict.get('K_Cross_Crop_Overlap', 'N/A')} |
| L. Avg Per-Crop Coverage | {quality_dict.get('L_Avg_PerCrop_Coverage', 'N/A')} |
| **Synthetic Quality Score (SQS)** | **{quality_dict.get('SQS_Score', 'N/A')} / 100** |

---

## D. Dependency Analysis Summary

- Dependency threshold used: |Spearman rho| >= {DEPENDENCY_THRESHOLD}
- Source: {n_total_pairs} unique feature pairs across 49 crop specification midpoints
- Retained dependencies: {n_retained} / {n_total_pairs}
- Features with dependencies preserved: see S4_dependency_analysis.xlsx

**Note**: With only 49 specification rows (one per crop), correlation estimates have
wide confidence intervals (approximately ±0.28 for n=49 at p=0.05). Conservative
thresholding at 0.25 is appropriate. The S4 framework does NOT claim these represent
true agronomic correlations — only that they are consistent with the available specification.

---

## E. Generation Summary

- Total crops processed: {len(df_summary)}
- Total synthetic samples: {len(df_summary) * N_SAMPLES_PER_CROP}
- Samples per crop: {N_SAMPLES_PER_CROP}
- Oversampling factor: {OVERSAMPLING_FACTOR}× ({OVERSAMPLING_FACTOR * N_SAMPLES_PER_CROP} candidates per crop)
- Selection method: Greedy Maximin in normalized 8D space × Joint Feasibility Score
- Total execution time: {elapsed_total:.1f} seconds

---

## F. Limitations

1. **Range-only source**: The source dataset provides only min/max bounds per crop, not
   observational measurements. S4 cannot recover information that doesn't exist in the source.

2. **49-point correlation estimates**: With only 49 crops, Spearman correlation estimates
   for 28 feature pairs are imprecise. Conservative thresholding mitigates fabrication risk.

3. **No causal model**: S4 does not model agronomic causal relationships — only statistical
   patterns observable in the specification table.

4. **Integer rounding**: Rounding to integers (TEMP, CROPDURATION, etc.) reduces diversity,
   especially for crops with narrow ranges. Post-rounding jitter is applied conservatively.

---

*Report generated automatically by S4_framework.py. Total execution time: {elapsed_total:.1f}s*
"""

    with open(OUTPUT_CONCLUSION_REPORT, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n[OK] {OUTPUT_CONCLUSION_REPORT}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    t_total_start = time.time()

    print("=" * 80)
    print("  S4 — JOINT CONSTRAINT AND BOUNDARY-AWARE SYNTHETIC DATA GENERATION")
    print("=" * 80)
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Samples per crop: {N_SAMPLES_PER_CROP}")
    print(f"  Oversampling factor: {OVERSAMPLING_FACTOR}× = {OVERSAMPLING_FACTOR * N_SAMPLES_PER_CROP} candidates/crop")
    print(f"  Dependency threshold: |rho| >= {DEPENDENCY_THRESHOLD}")
    print(f"  Near-duplicate threshold: {NEAR_DUPLICATE_THRESHOLD} (8D normalized)")
    print()

    # Step 1: Load source specification
    df_real = load_source_specification()
    n_crops = df_real['CROPS'].nunique()
    print(f"[S4] {n_crops} unique crop classes found.")

    # Step 2: Analyze crop ranges
    df_analysis, feature_mids, feature_spans = analyze_crop_ranges(df_real)

    # Step 4: Build dependency structure
    corr_matrix, df_dep, corr_raw = build_dependency_structure(df_real, feature_mids)

    # Step 13: Candidate strategy comparison (on first 8 crops for efficiency)
    print("\n[S4] Running candidate strategy comparison on sample crops...")
    sample_crops = df_real['CROPS'].unique()[:8]
    df_cand_comp, df_cand_agg = compare_candidate_strategies(
        df_real, df_analysis, corr_matrix, sample_crop_names=sample_crops
    )

    # Step 11: Full dataset generation
    df_final, df_summary = generate_s4_dataset(df_real, df_analysis, corr_matrix)

    # Step 12: Quality analysis
    quality_dict = analyze_quality(df_final, df_real, corr_matrix, label="S4")

    # Step 14: Joint space analysis
    df_joint = analyze_joint_space(df_final, df_real)

    # Step 15: Save outputs
    save_outputs(
        df_final, df_summary, df_analysis, df_dep, corr_raw,
        df_cand_comp, df_joint, quality_dict
    )

    # Step 16: Generate plots
    generate_plots(df_real, df_final, df_joint, df_cand_comp, quality_dict, corr_matrix)

    # Step 17: Generate conclusion report
    elapsed_total = time.time() - t_total_start
    generate_report(df_real, df_analysis, df_dep, "", quality_dict, df_summary, elapsed_total)

    # Final summary
    print("\n" + "=" * 80)
    print("  S4 EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"  Total rows generated: {len(df_final)}")
    print(f"  Total crops: {n_crops}")
    print(f"  SQS Score: {quality_dict.get('SQS_Score', 'N/A')} / 100")
    print(f"  OOB Rate: {quality_dict.get('A_OOB_Rate_%', 'N/A')}%")
    print(f"  Exact Duplicates: {quality_dict.get('H_Exact_Dup_Rate_%', 'N/A')}%")
    print(f"  Execution time: {elapsed_total:.1f}s")
    print("=" * 80)
    print(f"\n  Output files saved to: {DATASET_DIR}")
    for fname in [
        'S4_synthetic_dataset.xlsx',
        'S4_feature_analysis.xlsx',
        'S4_dependency_analysis.xlsx',
        'S4_candidate_comparison.xlsx',
        'S4_joint_space_analysis.xlsx',
        'S4_quality_analysis.xlsx',
        'S4_generation_summary.xlsx',
        'S4_conclusion_report.md',
    ]:
        print(f"    {fname}")


if __name__ == "__main__":
    main()
