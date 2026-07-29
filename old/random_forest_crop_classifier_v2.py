"""
Random Forest Classifier for Crop Recommendation System
==========================================================
Input dataset expected columns (edit COLUMN NAMES below to match your Excel exactly):
    N, P, K, temperature, humidity, ph, rainfall, label

'label' = crop name (49 classes: rice, samai, etc.)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -----------------------------
# 1. LOAD DATA
# -----------------------------
DATA_PATH = "Synthetic_Crop_Data_Beta_FInal.xlsx"   # <-- change if filename differs
LABEL_COL = "CROPS"              # <-- your crop-name column

# Columns that are dates/free text and add no predictive value as-is
DROP_COLS = ["SOWN", "HARVESTED"]

# Categorical columns to one-hot encode (edit if your sheet has more/fewer)
CATEGORICAL_COLS = ["TYPE_OF_CROP", "SOIL", "SEASON", "WATER_SOURCE"]

def load_data(path=DATA_PATH):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()  # remove stray whitespace in headers
    return df

# -----------------------------
# 2. PREPROCESS
# -----------------------------
def preprocess(df, label_col=LABEL_COL):
    df = df.dropna(subset=[label_col])          # drop rows with no crop label
    df = df.drop_duplicates()

    y = df[label_col]
    X = df.drop(columns=[label_col])

    # Drop columns with no predictive value (dates, free text, fully-empty cols)
    X = X.drop(columns=[c for c in DROP_COLS if c in X.columns])
    X = X.dropna(axis=1, how="all")  # drop fully-empty columns automatically

    # One-hot encode categorical columns instead of silently dropping them
    cat_cols_present = [c for c in CATEGORICAL_COLS if c in X.columns]
    if cat_cols_present:
        X = pd.get_dummies(X, columns=cat_cols_present, drop_first=False)
        # get_dummies produces bool dtype columns -- convert to int so they
        # aren't mistaken for "unhandled non-numeric" columns below
        bool_cols = X.select_dtypes(include="bool").columns
        X[bool_cols] = X[bool_cols].astype(int)

    # Remaining non-numeric columns (unexpected text columns) get dropped safely
    non_numeric_leftover = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_leftover:
        print(f"Warning: dropping unhandled non-numeric columns: {non_numeric_leftover}")
        X = X.drop(columns=non_numeric_leftover)

    X = X.fillna(X.mean(numeric_only=True))     # simple imputation for stray NaNs

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le

# -----------------------------
# 3. TRAIN / TEST SPLIT
# -----------------------------
def split_data(X, y, test_size=0.4, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

# -----------------------------
# 4. TRAIN RANDOM FOREST
# -----------------------------
def train_random_forest(X_train, y_train, tune=False):
    if tune:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"]
        }
        base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid = GridSearchCV(
            base_rf, param_grid, cv=5, scoring="accuracy",
            n_jobs=-1, verbose=1
        )
        grid.fit(X_train, y_train)
        print("Best params:", grid.best_params_)
        return grid.best_estimator_
    else:
        rf = RandomForestClassifier(
            n_estimators=10,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"   # helps if some crops have fewer samples
        )
        rf.fit(X_train, y_train)
        return rf

# -----------------------------
# 5. EVALUATE
# -----------------------------
def evaluate(model, X_test, y_test, label_encoder, plot=True):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted")

    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}\n")

    print("Classification Report:\n")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    if plot:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            cm, annot=False, cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Random Forest - Confusion Matrix")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("rf_confusion_matrix.png", dpi=150)
        plt.close()

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# -----------------------------
# 6. FEATURE IMPORTANCE
# -----------------------------
def feature_importance(model, feature_names, plot=True):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)
    print("\nFeature Importances:\n", importances)

    if plot:
        plt.figure(figsize=(8, 5))
        importances.plot(kind="bar", color="seagreen")
        plt.title("Random Forest Feature Importance")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.savefig("rf_feature_importance.png", dpi=150)
        plt.close()

    return importances

# -----------------------------
# 7. CROSS-VALIDATION CHECK
# -----------------------------
def cross_validate(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"\n{cv}-Fold CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores

# -----------------------------
# 8. SAVE MODEL
# -----------------------------
def save_model(model, label_encoder, model_path="rf_crop_model.pkl", le_path="label_encoder.pkl"):
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, le_path)
    print(f"\nModel saved to {model_path}")
    print(f"Label encoder saved to {le_path}")

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main(tune=False):
    df = load_data()
    X, y, le = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_random_forest(X_train, y_train, tune=tune)

    print("\n===== TEST SET RESULTS =====")
    evaluate(model, X_test, y_test, le)

    print("\n===== CROSS-VALIDATION (full data) =====")
    cross_validate(model, X, y)

    feature_importance(model, X.columns.tolist())

    save_model(model, le)

    return model, le, X, y

if __name__ == "__main__":
    main(tune=False)