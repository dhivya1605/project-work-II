# 9_evaluate_synthetic_techniques_tstr.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ========================== CONFIGURATION ==========================
RANDOM_STATE = 42
TEST_SAMPLES_PER_CROP = 20
TARGET = 'CROPS'
REAL_DATASET_FILE = 'crop-dataset.xlsx'

# All 8 generated synthetic datasets
TECHNIQUE_FILES = {
    "T1_Sobol": "1_synthetic_crop_data_sobol.xlsx",
    "T2_TruncNorm": "2_synthetic_crop_data_truncnorm.xlsx",
    # "T3_Beta_Crop": "3_Synthetic_crop_data_beta_crop_specific.xlsx",
    "T4_Beta_Final": "4_Synthetic_Crop_Data_Beta_Final.xlsx",
    "T5_Copula": "5_Synthetic_Crop_Data_Copula_Final.xlsx",
    # "T6_Best_Copula": "6_Synthetic_Crop_Data_Best_Copula.xlsx",
    "T7_AHAPSF": "7_Synthetic_Crop_Data_AHAPSF.xlsx",
    # "T8_Hybrid_Final": "8_Synthetic_Crop_Data_Hybrid_Final.xlsx",
    "T9_LHS": "f_synthetic_crop_data_lhs.xlsx",
}

NUMERIC_FEATURES = ['SOIL_PH', 'CROPDURATION', 'TEMP', 'WATERREQUIRED',
                    'RELATIVE_HUMIDITY', 'N', 'P', 'K']

# ========================== CREATE REAL TEST SET (Strict Ranges) ==========================
print(f"Creating Real Test Set from {REAL_DATASET_FILE} (per crop min/max ranges)...")

df_real = pd.read_excel(REAL_DATASET_FILE)
df_real.columns = [col.strip().replace('\n', ' ').replace(' ', '_')
                   .replace('(', '').replace(')', '').replace('/', '_').upper()
                   for col in df_real.columns]

np.random.seed(RANDOM_STATE)
test_rows = []

for _, row in df_real.iterrows():
    crop = row[TARGET]

    # Robustly get min/max values, handling NaNs and ensuring min < max
    def get_range(min_col_name, max_col_name, default_min, default_max, epsilon=0.1):
        min_val = row.get(min_col_name)
        max_val = row.get(max_col_name)

        if pd.isna(min_val): min_val = default_min
        if pd.isna(max_val): max_val = default_max

        min_val, max_val = float(min_val), float(max_val)
        # Ensure min is strictly less than max
        if min_val >= max_val: max_val = min_val + epsilon
        return min_val, max_val

    soil_ph_low, soil_ph_high = get_range('SOIL_PH_LOW', 'SOIL_PH_HIGH', 0.0, 14.0, 0.1)
    cropduration_min, cropduration_max = get_range('CROPDURATION_MIN', 'CROPDURATION_MAX', 0.0, 365.0, 1.0)
    min_temp, max_temp = get_range('MIN_TEMP', 'MAX_TEMP', 0.0, 50.0, 1.0)
    waterrequired_min, waterrequired_max = get_range('WATERREQUIRED_MIN', 'WATERREQUIRED_MAX', 0.0, 5000.0, 1.0)
    relative_humidity_min, relative_humidity_max = get_range('RELATIVE_HUMIDITY_MIN', 'RELATIVE_HUMIDITY_MAX', 0.0, 100.0, 1.0)
    n_min, n_max = get_range('N_MIN', 'N_MAX', 0.0, 200.0, 1.0)
    p_min, p_max = get_range('P_MIN', 'P_MAX', 0.0, 100.0, 1.0)
    k_min, k_max = get_range('K_MIN', 'K_MAX', 0.0, 100.0, 1.0)

    for _ in range(TEST_SAMPLES_PER_CROP):
        sample = {
            'CROPS': crop,
            'SOIL_PH': np.random.uniform(soil_ph_low, soil_ph_high),
            'CROPDURATION': np.random.uniform(cropduration_min, cropduration_max),
            'TEMP': np.random.uniform(min_temp, max_temp),
            'WATERREQUIRED': np.random.uniform(waterrequired_min, waterrequired_max),
            'RELATIVE_HUMIDITY': np.random.uniform(relative_humidity_min, relative_humidity_max),
            'N': np.random.uniform(n_min, n_max),
            'P': np.random.uniform(p_min, p_max),
            'K': np.random.uniform(k_min, k_max)
        }
        test_rows.append(sample)

df_test = pd.DataFrame(test_rows)
print(f"Real test set created: {len(df_test)} samples ({TEST_SAMPLES_PER_CROP} per crop)\n")

# Fit LabelEncoder on the full list of crops from df_real
label_encoder = LabelEncoder()
label_encoder.fit(df_real['CROPS'].astype(str).str.strip().unique())

y_test = label_encoder.transform(df_test['CROPS'].astype(str).str.strip())
X_test = df_test[NUMERIC_FEATURES].values.astype(np.float32)

# Impute NaNs in X_test if any exist after generation
if np.isnan(X_test).any():
    print("   Warning: NaNs found in X_test. Imputing with column mean.")
    imputer_test = SimpleImputer(strategy='mean')
    X_test = imputer_test.fit_transform(X_test)


# ========================== TSTR EVALUATION ==========================
print("Starting TSTR Evaluation using ML Models...\n")

final_results = {model: {} for model in ["RF", "XGBoost", "SVM", "kNN"]}

for tech_key, train_file in TECHNIQUE_FILES.items():
    print(f"Evaluating {tech_key} ({train_file}) ...")

    try:
        df_train = pd.read_excel(train_file)
    except FileNotFoundError:
        print(f"   ❌ File not found: {train_file}. Skipping {tech_key}.")
        continue

    df_train.columns = [col.strip().replace('\n', ' ').replace(' ', '_')
                        .replace('(', '').replace(')', '').replace('/', '_').upper()
                        for col in df_train.columns]

    X_train = df_train[NUMERIC_FEATURES].values.astype(np.float32)

    # Impute NaNs in X_train if any exist
    if np.isnan(X_train).any():
        print(f"   Warning: NaNs found in X_train for {tech_key}. Imputing with column mean.")
        imputer_train = SimpleImputer(strategy='mean')
        X_train = imputer_train.fit_transform(X_train)

    y_train = label_encoder.transform(df_train['CROPS'].astype(str).str.strip())

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "RF": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, random_state=RANDOM_STATE, eval_metric='mlogloss'),
        "SVM": SVC(C=10, kernel='rbf', random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=7)
    }

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)

        final_results[model_name][tech_key] = {
            'Accuracy': round(accuracy_score(y_test, pred), 4),
            'Precision': round(precision_score(y_test, pred, average='macro', zero_division=0), 4),
            'Recall': round(recall_score(y_test, pred, average='macro', zero_division=0), 4),
            'F1': round(f1_score(y_test, pred, average='macro', zero_division=0), 4)
        }

    print(f"   ✓ {tech_key} completed.")

# ========================== PRINT TABLES ==========================
print("\n" + "="*140)
print("TSTR RESULTS - Train on Synthetic, Test on Real (ML Models)")
print("="*140)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
tech_keys = list(TECHNIQUE_FILES.keys())

for metric in metrics:
    print(f"\n{metric.upper()} (Macro Average)")
    print("-" * 140)
    header = "Model".ljust(12) + "".join([t.rjust(16) for t in tech_keys])
    print(header)
    print("-" * 140)

    for model_name in ["RF", "XGBoost", "SVM", "kNN"]:
        row = [model_name.ljust(12)]
        for t in tech_keys:
            val = final_results[model_name].get(t, {}).get(metric, np.nan)
            row.append(f"{val:16.4f}" if not np.isnan(val) else "N/A".rjust(16))
        print("".join(row))

# Save results to Excel
results_list = []
for model_name in ["RF", "XGBoost", "SVM", "kNN"]:
    for metric in metrics:
        row_data = {'Model_Metric': f"{model_name}_{metric}"}
        for t in tech_keys:
            val = final_results[model_name].get(t, {}).get(metric, np.nan)
            row_data[t] = val
        results_list.append(row_data)

df_save = pd.DataFrame(results_list).set_index('Model_Metric')
output_excel = "TSTR_ML_Results_All_Metrics.xlsx"
df_save.to_excel(output_excel)

print("\n" + "="*140)
print(f"✅ TSTR Evaluation Completed! Results saved to '{output_excel}'")
print("="*140)
