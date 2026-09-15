"""
train.py
--------
End-to-end training pipeline:

    dataset.csv
        -> SFOA (feature selection)
        -> DLO  (TabKANet hyperparameter tuning, on selected features)
        -> final TabKANet training (full epochs, best hyperparameters)
        -> evaluation + saved artifacts (model weights, scaler, label encoder,
           selected feature list) for your Flask + SHAP dashboard

USAGE
-----
1. Edit the CONFIG block below:
   - DATA_PATH: path to your synthetic/real dataset CSV
   - FEATURE_COLUMNS: your numeric agronomic columns
   - LABEL_COLUMN: your crop-name column
2. Run:  python train.py

This script prints progress from SFOA and DLO as they run, then trains and
saves the final model to ./artifacts/
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from sfoa import SFOAFeatureSelector
from dlo import DLOHyperparameterOptimizer
from tabkanet import TabKANet

# --------------------------------------------------------------------------- #
# CONFIG  -- edit these to match your dataset
# --------------------------------------------------------------------------- #
CONFIG = {
    "DATA_PATH": r"d:\crop-recommendation\dataset\10_synthetic_Crop_data_ahapsf1.xlsx", # <-- point this at your synthetic dataset CSV
    "FEATURE_COLUMNS": [
        "SOIL_PH", "N", "P", "K", "TEMP",
        "WATERREQUIRED", "RELATIVE_HUMIDITY", "CROPDURATION",
    ],
    "LABEL_COLUMN": "CROPS",

    # --- SFOA: was 20 pop × 30 iters = ~740 RF fits (~2-3 hrs on CPU) ---
    # Now  10 pop × 15 iters = ~165 RF fits  (~20-30 min on CPU)
    "SFOA_POPULATION": 10,
    "SFOA_ITERATIONS": 15,

    # --- DLO: was 12 pop × 15 iters × 8 probe epochs = 1,536 epoch-runs ---
    # Now  8 pop  × 8  iters × 5 probe epochs = 360  epoch-runs  (~5-10 min)
    "DLO_POPULATION": 8,
    "DLO_ITERATIONS": 8,
    "DLO_PROBE_EPOCHS": 5,      # quick training epochs used only during DLO search

    "FINAL_EPOCHS": 80,         # unchanged — full quality final training
    "BATCH_SIZE": 64,
    "TEST_SIZE": 0.2,
    "VAL_SIZE": 0.1,            # taken out of the training split
    "RANDOM_STATE": 42,
    "OUTPUT_DIR": "artifacts",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
def load_data(cfg):
    path = cfg["DATA_PATH"]
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df.columns = [col.strip() for col in df.columns]
    X = df[cfg["FEATURE_COLUMNS"]].values.astype(np.float32)
    y_raw = df[cfg["LABEL_COLUMN"]].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    return X, y, label_encoder


def train_tabkanet(
    X_train, y_train, X_val, y_val,
    num_features, num_classes, hyperparams, epochs, batch_size, verbose=False,
):
    model = TabKANet(
        num_features=num_features,
        num_classes=num_classes,
        embed_dim=hyperparams["embed_dim"],
        n_heads=hyperparams["n_heads"],
        n_transformer_layers=hyperparams["n_transformer_layers"],
        ff_dim=hyperparams["ff_dim"],
        dropout=hyperparams["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams["learning_rate"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()
        best_val_acc = max(best_val_acc, val_acc)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch + 1:03d}/{epochs} val_acc={val_acc:.4f}")

    return model, best_val_acc


def main():
    cfg = CONFIG
    os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)

    print("=" * 70)
    print("STEP 1/4: Loading data")
    print("=" * 70)
    X_all, y_all, label_encoder = load_data(cfg)
    print(f"Loaded {X_all.shape[0]} rows, {X_all.shape[1]} raw features, "
          f"{len(label_encoder.classes_)} classes")

    # Train / test split up front so test set never touches feature selection,
    # HPO, or final training
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_all, y_all, test_size=cfg["TEST_SIZE"],
        random_state=cfg["RANDOM_STATE"], stratify=y_all,
    )

    print("\n" + "=" * 70)
    print("STEP 2/4: SFOA feature selection")
    print("=" * 70)
    sfoa = SFOAFeatureSelector(
        population_size=cfg["SFOA_POPULATION"],
        max_iterations=cfg["SFOA_ITERATIONS"],
        random_state=cfg["RANDOM_STATE"],
    )
    sfoa.fit(X_trainval, y_trainval, verbose=True)
    selected_idx = sfoa.get_selected_indices()
    if len(selected_idx) == 0:
        print("WARNING: SFOA selected 0 features! Defaulting to all features.")
        selected_idx = np.arange(len(cfg["FEATURE_COLUMNS"]))
    selected_columns = [cfg["FEATURE_COLUMNS"][i] for i in selected_idx]
    print(f"\nSFOA selected {len(selected_idx)}/{len(cfg['FEATURE_COLUMNS'])} features: {selected_columns}")

    X_trainval_sel = X_trainval[:, selected_idx]
    X_test_sel = X_test[:, selected_idx]

    # Scale AFTER feature selection, fit scaler on train+val only
    scaler = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval_sel).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_sel).astype(np.float32)

    # Split off a validation set for HPO + final training early stopping proxy
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval_scaled, y_trainval, test_size=cfg["VAL_SIZE"],
        random_state=cfg["RANDOM_STATE"], stratify=y_trainval,
    )

    num_features = X_train.shape[1]
    num_classes = len(label_encoder.classes_)

    print("\n" + "=" * 70)
    print("STEP 3/4: DLO hyperparameter optimization for TabKANet")
    print("=" * 70)

    def dlo_fitness(hyperparams):
        # Short probe-training run; fitness = 1 - val_accuracy (lower is better)
        _, val_acc = train_tabkanet(
            X_train, y_train, X_val, y_val,
            num_features, num_classes, hyperparams,
            epochs=cfg["DLO_PROBE_EPOCHS"], batch_size=cfg["BATCH_SIZE"],
            verbose=False,
        )
        return 1.0 - val_acc

    dlo = DLOHyperparameterOptimizer(
        population_size=cfg["DLO_POPULATION"],
        max_iterations=cfg["DLO_ITERATIONS"],
        random_state=cfg["RANDOM_STATE"],
    )
    best_hyperparams = dlo.optimize(dlo_fitness, verbose=True)
    print(f"\nDLO best hyperparameters: {best_hyperparams}")

    print("\n" + "=" * 70)
    print("STEP 4/4: Final TabKANet training with DLO-tuned hyperparameters")
    print("=" * 70)
    final_model, final_val_acc = train_tabkanet(
        X_train, y_train, X_val, y_val,
        num_features, num_classes, best_hyperparams,
        epochs=cfg["FINAL_EPOCHS"], batch_size=cfg["BATCH_SIZE"],
        verbose=True,
    )
    print(f"Final validation accuracy: {final_val_acc:.4f}")

    # ---- Test set evaluation ----
    final_model.eval()
    with torch.no_grad():
        test_logits = final_model(torch.tensor(X_test_scaled).to(DEVICE))
        test_preds = test_logits.argmax(dim=1).cpu().numpy()

    test_acc = accuracy_score(y_test, test_preds)
    test_prec_weighted = precision_score(y_test, test_preds, average="weighted", zero_division=0)
    test_rec_weighted = recall_score(y_test, test_preds, average="weighted", zero_division=0)
    test_f1_weighted = f1_score(y_test, test_preds, average="weighted", zero_division=0)

    test_prec_macro = precision_score(y_test, test_preds, average="macro", zero_division=0)
    test_rec_macro = recall_score(y_test, test_preds, average="macro", zero_division=0)
    test_f1_macro = f1_score(y_test, test_preds, average="macro", zero_division=0)

    print("\n" + "=" * 70)
    print("HELD-OUT TEST SET EVALUATION METRICS")
    print("=" * 70)
    print(f"  Test Accuracy:        {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"  Precision (Weighted): {test_prec_weighted:.4f}")
    print(f"  Recall (Weighted):    {test_rec_weighted:.4f}")
    print(f"  F1-Score (Weighted):  {test_f1_weighted:.4f}")
    print(f"  Precision (Macro):    {test_prec_macro:.4f}")
    print(f"  Recall (Macro):       {test_rec_macro:.4f}")
    print(f"  F1-Score (Macro):     {test_f1_macro:.4f}")

    # ---- Save everything the dashboard needs ----
    out_dir = cfg["OUTPUT_DIR"]
    torch.save(final_model.state_dict(), os.path.join(out_dir, "tabkanet_weights.pt"))

    import joblib
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(out_dir, "label_encoder.pkl"))

    with open(os.path.join(out_dir, "model_config.json"), "w") as f:
        json.dump({
            "selected_columns": selected_columns,
            "num_features": num_features,
            "num_classes": num_classes,
            "hyperparameters": best_hyperparams,
            "test_accuracy": float(test_acc),
            "test_precision": float(test_prec_weighted),
            "test_recall": float(test_rec_weighted),
            "test_f1_score": float(test_f1_weighted),
            "test_precision_macro": float(test_prec_macro),
            "test_recall_macro": float(test_rec_macro),
            "test_f1_score_macro": float(test_f1_macro),
            "final_val_accuracy": float(final_val_acc),
        }, f, indent=2)

    print(f"\nSaved model, scaler, label encoder, and config to ./{out_dir}/")
    print("Use model_config.json to rebuild TabKANet(**hyperparameters) for inference/SHAP.")


if __name__ == "__main__":
    main()
