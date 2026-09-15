"""
run_tabkanet_standalone.py
--------------------------
Standalone script to train, evaluate, predict, and explain using the TabKANet Model
(KAN Feature Embedding + Transformer Encoder Self-Attention Backbone).
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shap

from tabkanet import TabKANet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("=" * 70)
    print(" 🎯 TABKANET MODEL: TRAINING, PREDICTION & EXPLAINABILITY")
    print("=" * 70)

    data_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "7_Synthetic_Crop_Data_AHAPSF.xlsx")
    if not os.path.exists(data_path):
        print(f"Error: Dataset file not found at {data_path}")
        return

    print(f"📂 Loading dataset: {os.path.basename(data_path)} ...")
    df = pd.read_excel(data_path)
    df.columns = [col.strip().upper() for col in df.columns]

    feature_cols = ["SOIL_PH", "N", "P", "K", "TEMP", "WATERREQUIRED", "RELATIVE_HUMIDITY", "CROPDURATION"]
    avail_cols = [c for c in feature_cols if c in df.columns]

    X_raw = df[avail_cols].values.astype(np.float32)
    le = LabelEncoder()
    y_raw = le.fit_transform(df["CROPS"].values)

    num_features = len(avail_cols)
    num_classes = len(le.classes_)

    # Split & Normalize
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    train_ds = TensorDataset(torch.tensor(X_train_s), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test_s), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    print(f"Dataset summary: {len(df)} samples | {num_features} Features | {num_classes} Crop Classes")

    # 1. Train TabKANet
    print("\n🚀 Step 1: Training TabKANet Model (KAN + Transformer Backbone) ...")
    model = TabKANet(
        num_features=num_features,
        num_classes=num_classes,
        embed_dim=32,
        num_grids=8,
        n_heads=4,
        n_transformer_layers=2,
        ff_dim=64,
        dropout=0.1
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 31):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    print("✅ Training Complete!")

    # 2. Evaluation Metrics
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.tensor(X_test_s).to(DEVICE))
        preds = test_logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    print("\n📊 TabKANet Performance Metrics:")
    print(f"  • Accuracy        : {acc * 100:.2f}%")
    print(f"  • Precision (W)   : {prec * 100:.2f}%")
    print(f"  • Recall (W)      : {rec * 100:.2f}%")
    print(f"  • F1-Score (W)    : {f1 * 100:.2f}%")

    # 3. Sample Prediction
    print("\n🔮 Step 2: Running Sample Prediction with TabKANet ...")
    sample_input = np.array([[6.5, 80.0, 45.0, 40.0, 27.5, 1200.0, 70.0, 120.0]], dtype=np.float32)
    sample_s = scaler.transform(sample_input).astype(np.float32)
    
    with torch.no_grad():
        out_logits = model(torch.tensor(sample_s).to(DEVICE))
        probs = torch.softmax(out_logits, dim=-1).cpu().numpy()[0]
        top_idx = probs.argmax()
        top_crop = le.inverse_transform([top_idx])[0]

    print(f"  🎯 Recommended Crop  : {top_crop.upper()}")
    print(f"  📊 Confidence Score  : {probs[top_idx] * 100:.2f}%")

    # 4. SHAP Explanation
    print("\n🧠 Step 3: Computing SHAP Feature Attribution for TabKANet ...")
    def model_predict_func(x_numpy):
        x_torch = torch.tensor(x_numpy.astype(np.float32)).to(DEVICE)
        with torch.no_grad():
            out = model(x_torch)
            return torch.softmax(out, dim=-1).cpu().numpy()

    explainer = shap.KernelExplainer(model_predict_func, X_train_s[:30])
    shap_vals = explainer.shap_values(sample_s, l1_reg=False)

    print("\nFeature Impact on Prediction:")
    # Handle SHAP multi-class output shape
    if isinstance(shap_vals, list):
        class_shap = shap_vals[top_idx][0]
    elif shap_vals.ndim == 3:
        class_shap = shap_vals[0, :, top_idx]
    else:
        class_shap = shap_vals[0]

    for col_name, score in zip(avail_cols, class_shap):
        direction = "Positive (+)" if score > 0 else "Negative (-)"
        print(f"  • {col_name:20s}: {score:+.4f} ({direction})")

    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    main()
