# f_Synthetic_crop_data_lhs.py
# Latin Hypercube Sampling (LHS) for Synthetic Crop Dataset Generation

import pandas as pd
import numpy as np

# ========================= CONFIGURATION =========================
N_SAMPLES_PER_CROP = 600
EXCEL_FILE = 'crop-dataset.xlsx'
OUTPUT_EXCEL = 'f_synthetic_crop_data_lhs.xlsx'
RANDOM_SEED = 42


def generate_lhs_synthetic_data():
    """
    Generates synthetic crop dataset using Latin Hypercube Sampling (LHS)
    following the pseudocode:
    
    For each crop c in C:
      Extract crop-specific feature ranges [MIN_{c,f}, MAX_{c,f}].
      For each numerical feature f in F:
        1. R_{c,f} = MAX_{c,f} - MIN_{c,f}
        2. Stratum_width = R_{c,f} / n_c
        3. For sample i = 1 to n_c:
             L_i = MIN_{c,f} + (i - 1) * Stratum_width
             U_i = MIN_{c,f} + i * Stratum_width
             Sample_i = RandomUniform(L_i, U_i)
        4. Shuffle samples independently using a random permutation.
      Construct sample vector s_i = (x_1, x_2, ..., x_d) for each sample i.
    """
    df_real = pd.read_excel(EXCEL_FILE)
    df_real.columns = [col.strip() for col in df_real.columns]

    crop_list = df_real['CROPS'].astype(str).str.strip().unique()
    synthetic_samples = []

    print(f"Generating LHS synthetic samples ({N_SAMPLES_PER_CROP} per crop)...\n")

    # Mappings from target feature names to min/max columns in crop-dataset.xlsx
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

    for crop_idx, crop in enumerate(crop_list):
        df_c = df_real[df_real['CROPS'].astype(str).str.strip() == crop]
        if df_c.empty:
            continue
        row = df_c.iloc[0]

        # Step 1: Extract crop-specific feature ranges
        feature_ranges = {}
        for feat_name, (min_col, max_col) in mappings.items():
            if min_col in row and max_col in row:
                a = float(row[min_col])
                b = float(row[max_col])
                if pd.isna(a) or pd.isna(b):
                    continue
                if a >= b:
                    b = a + (0.1 if feat_name == 'SOIL_PH' else 1.0)
                feature_ranges[feat_name] = (a, b)

        features = list(feature_ranges.keys())
        if not features:
            continue

        # Random generator per crop for reproducibility
        rng = np.random.default_rng(RANDOM_SEED + crop_idx)

        crop_feature_samples = {}

        # Step 2: For each numerical feature f in F
        for feat in features:
            min_val, max_val = feature_ranges[feat]

            # Compute feature range: R_{c,f} = MAX_{c,f} - MIN_{c,f}
            R_cf = max_val - min_val

            # Divide interval into n_c equally probable strata
            stratum_width = R_cf / N_SAMPLES_PER_CROP

            feat_samples = np.zeros(N_SAMPLES_PER_CROP)

            # Generate one sample per stratum: L_i to U_i
            for i in range(1, N_SAMPLES_PER_CROP + 1):
                L_i = min_val + (i - 1) * stratum_width
                U_i = min_val + i * stratum_width
                feat_samples[i - 1] = rng.uniform(L_i, U_i)

            # Shuffle samples independently using a random permutation
            rng.shuffle(feat_samples)

            crop_feature_samples[feat] = feat_samples

        # Step 3: Construct sample vectors s_i = (x_1, x_2, ..., x_d)
        crop_df = pd.DataFrame(crop_feature_samples)
        crop_df['CROPS'] = crop

        # Add categorical metadata
        cat_cols = ['TYPE_OF_CROP', 'SOIL', 'SEASON', 'SOWN', 'HARVESTED', 'WATER_SOURCE']
        for col in cat_cols:
            if col in row:
                crop_df[col] = row[col]

        # Step 4: Rounding and data formatting
        if 'SOIL_PH' in crop_df.columns:
            crop_df['SOIL_PH'] = crop_df['SOIL_PH'].round(2)
        for col in ['TEMP', 'CROPDURATION', 'WATERREQUIRED', 'RELATIVE_HUMIDITY', 'N', 'P', 'K']:
            if col in crop_df.columns:
                crop_df[col] = crop_df[col].round(0).astype(int)

        synthetic_samples.append(crop_df)
        print(f"✓ Generated {N_SAMPLES_PER_CROP} LHS samples for: {crop}")

    # Combine all crops into final dataset
    final_df = pd.concat(synthetic_samples, ignore_index=True)

    # Standard column order
    final_order = ['CROPS', 'TYPE_OF_CROP', 'SOIL', 'SEASON', 'SOWN', 'HARVESTED',
                   'WATER_SOURCE', 'SOIL_PH', 'TEMP', 'CROPDURATION',
                   'WATERREQUIRED', 'RELATIVE_HUMIDITY', 'N', 'P', 'K']
    final_df = final_df[[col for col in final_order if col in final_df.columns]]

    print(f"\n✅ LHS Generation Completed! Total samples generated: {len(final_df)}")
    return final_df


# ========================= EXECUTION =========================
if __name__ == "__main__":
    df_synth = generate_lhs_synthetic_data()
    df_synth.to_excel(OUTPUT_EXCEL, index=False)
    print(f"📁 Dataset saved as: {OUTPUT_EXCEL}")
