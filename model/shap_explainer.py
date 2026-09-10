"""
shap_explainer.py
-----------------
SHAP explainability module for TabKANet Crop Recommendation model.

Uses model-agnostic `shap.KernelExplainer` to compute local feature attributions
for any input query. Operates on raw agricultural features and applies the exact
inference preprocessing pipeline inside the prediction wrapper.
"""

import os
import sys
import numpy as np
import pandas as pd

# Set headless matplotlib backend before any plot module imports
import matplotlib
matplotlib.use('Agg')

import torch
import shap
from predict import CropPredictor, FEATURE_DISPLAY_NAMES, DEFAULT_FEATURE_VALUES

ALL_FEATURE_COLUMNS = [
    "N", "P", "K", "SOIL_PH", "WATERREQUIRED", "TEMP", "RELATIVE_HUMIDITY", "CROPDURATION"
]


PRIMARY_DISPLAY_FEATURES = [
    "N", "P", "K", "SOIL_PH", "WATERREQUIRED", "TEMP"
]


class CropSHAPExplainer:
    def __init__(self, predictor: CropPredictor, background_df: pd.DataFrame = None):
        self.predictor = predictor
        self.background_df = background_df if background_df is not None else self._load_default_background_data()
        self.explainer = None
        self._init_explainer()

    def _load_default_background_data(self) -> pd.DataFrame:
        """
        Loads background sample dataset from dataset folder or generates
        representative agronomic sampling.
        """
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "dataset", "7_Synthetic_Crop_Data_AHAPSF.xlsx"),
            os.path.join(os.path.dirname(__file__), "..", "dataset", "crop-dataset.xlsx"),
            os.path.join(os.path.dirname(__file__), "..", "dataset", "1_synthetic_crop_data_sobol.xlsx"),
        ]

        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                try:
                    df = pd.read_excel(abs_path)
                    df.columns = [col.strip().upper() for col in df.columns]
                    # Ensure column aliases
                    aliases = {
                        "RAINFALL": "WATERREQUIRED", "TEMP": "TEMP", "PH": "SOIL_PH"
                    }
                    for alias, target in aliases.items():
                        if alias in df.columns and target not in df.columns:
                            df[target] = df[alias]
                    
                    cols_to_keep = [col for col in ALL_FEATURE_COLUMNS if col in df.columns]
                    if len(cols_to_keep) >= 4:
                        # Sample 50 representative rows
                        sub_df = df[cols_to_keep].dropna().sample(n=min(60, len(df)), random_state=42)
                        for missing_col in ALL_FEATURE_COLUMNS:
                            if missing_col not in sub_df.columns:
                                sub_df[missing_col] = DEFAULT_FEATURE_VALUES[missing_col]
                        return sub_df[ALL_FEATURE_COLUMNS]
                except Exception as e:
                    print(f"Warning: Failed loading dataset from {abs_path}: {e}")

        # Fallback representative dataset
        np.random.seed(42)
        samples = 50
        data = {
            "N": np.random.uniform(10, 140, samples),
            "P": np.random.uniform(5, 90, samples),
            "K": np.random.uniform(10, 200, samples),
            "SOIL_PH": np.random.uniform(4.5, 8.5, samples),
            "WATERREQUIRED": np.random.uniform(300, 2500, samples),
            "TEMP": np.random.uniform(12, 38, samples),
            "RELATIVE_HUMIDITY": np.random.uniform(30, 95, samples),
            "CROPDURATION": np.random.uniform(60, 210, samples),
        }
        return pd.DataFrame(data)

    def _init_explainer(self):
        """
        Creates background sample summary and initializes shap.KernelExplainer.
        """
        # Take a summarized sample background of 40 instances to keep SHAP fast
        bg_sample = shap.sample(self.background_df[ALL_FEATURE_COLUMNS], 30)

        def model_predict_target_prob(X_raw):
            """
            Wrapper for KernelExplainer: takes raw X_raw array, preprocesses,
            returns class probability array.
            """
            if not isinstance(X_raw, pd.DataFrame):
                X_raw_df = pd.DataFrame(X_raw, columns=ALL_FEATURE_COLUMNS)
            else:
                X_raw_df = X_raw

            X_scaled = self.predictor.preprocess(X_raw_df)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.predictor.device)

            with torch.no_grad():
                logits = self.predictor.model(X_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        self.predict_fn = model_predict_target_prob
        self.bg_sample = bg_sample
        # KernelExplainer on model probability output
        self.explainer = shap.KernelExplainer(self.predict_fn, self.bg_sample, link="identity")

    def explain(self, input_dict: dict, target_class_idx: int = None, nsamples: int = 80) -> dict:
        """
        Computes local SHAP values for a single input query.

        Returns:
            dict with SHAP values, feature table, and human readable text summary.
        """
        input_df = self.predictor.prepare_input_dataframe(input_dict)[ALL_FEATURE_COLUMNS]

        if target_class_idx is None:
            pred_res = self.predictor.predict(input_df)
            target_class_idx = pred_res["predicted_class_index"]
            crop_name = pred_res["predicted_crop"]
            confidence_pct = pred_res["confidence_pct"]
        else:
            crop_name = self.predictor.label_encoder.inverse_transform([target_class_idx])[0]
            confidence_pct = 0.0

        # Calculate SHAP values for raw input
        # KernelExplainer outputs list of arrays (one per class) or 3D array
        raw_values = self.explainer.shap_values(input_df, nsamples=nsamples)

        if isinstance(raw_values, list):
            class_shap = raw_values[target_class_idx][0]
        elif isinstance(raw_values, np.ndarray) and raw_values.ndim == 3:
            class_shap = raw_values[0, :, target_class_idx]
        else:
            class_shap = np.array(raw_values).flatten()

        base_value = (
            self.explainer.expected_value[target_class_idx]
            if isinstance(self.explainer.expected_value, (list, np.ndarray))
            else self.explainer.expected_value
        )

        rows = []
        for i, col in enumerate(ALL_FEATURE_COLUMNS):
            val = float(input_df[col].iloc[0])
            shap_val = float(class_shap[i])
            impact = "Positive (+)" if shap_val > 1e-4 else ("Negative (-)" if shap_val < -1e-4 else "Neutral")
            rows.append({
                "Feature_Code": col,
                "Feature_Name": FEATURE_DISPLAY_NAMES.get(col, col),
                "User_Value": val,
                "SHAP_Value": shap_val,
                "SHAP_Abs": abs(shap_val),
                "Impact": impact,
            })

        shap_table = pd.DataFrame(rows).sort_values("SHAP_Abs", ascending=False).reset_index(drop=True)

        # Generate human readable dynamic explanation text
        pos_df = shap_table[shap_table["SHAP_Value"] > 1e-4]
        neg_df = shap_table[shap_table["SHAP_Value"] < -1e-4]

        pos_features = pos_df["Feature_Name"].tolist()
        neg_features = neg_df["Feature_Name"].tolist()

        explanation_parts = [f"The model recommended **{crop_name}** with **{confidence_pct}%** confidence."]
        
        if pos_features:
            explanation_parts.append(
                f"Among the input features, **{', '.join(pos_features[:3])}** had a **positive contribution** pushing the model toward this recommendation."
            )
        if neg_features:
            explanation_parts.append(
                f"Conversely, **{', '.join(neg_features[:2])}** had a **negative impact** against this prediction."
            )
        if not pos_features and not neg_features:
            explanation_parts.append("All input features contributed neutrally toward the baseline expectation.")

        explanation_text = " ".join(explanation_parts)

        return {
            "crop_name": crop_name,
            "target_class_index": target_class_idx,
            "base_value": float(base_value),
            "shap_table": shap_table,
            "raw_shap_values": class_shap,
            "feature_columns": ALL_FEATURE_COLUMNS,
            "explanation_text": explanation_text,
        }


def run_terminal_shap_cli():
    print("=" * 70)
    print("  🧠 TABKANET SHAP LOCAL EXPLAINABILITY MODULE (CLI)")
    print("=" * 70)
    print("Loading trained model artifacts & background reference dataset...")
    
    predictor = CropPredictor()
    explainer = CropSHAPExplainer(predictor)
    print("✅ Model & SHAP Explainer loaded successfully!\n")
    print("Please enter the agronomic & environmental values below.")
    print("(Press ENTER to accept the default value shown in brackets)\n")

    input_dict = {}
    prompts = [
        ("N", "Nitrogen (N) [mg/kg]", 50.0),
        ("P", "Phosphorus (P) [mg/kg]", 40.0),
        ("K", "Potassium (K) [mg/kg]", 40.0),
        ("SOIL_PH", "Soil pH (0-14)", 6.5),
        ("TEMP", "Temperature (°C)", 25.0),
        ("WATERREQUIRED", "Water / Rainfall (mm)", 1000.0),
        ("RELATIVE_HUMIDITY", "Relative Humidity (%)", 70.0),
        ("CROPDURATION", "Crop Duration (Days)", 120.0),
    ]

    for key, label, default_val in prompts:
        try:
            val_str = input(f"  ➜ {label} [{default_val}]: ").strip()
            if val_str == "":
                input_dict[key] = float(default_val)
            else:
                input_dict[key] = float(val_str)
        except ValueError:
            print(f"    ⚠️ Invalid input. Using default value: {default_val}")
            input_dict[key] = float(default_val)

    print("\n" + "─" * 70)
    print("Computing SHAP local feature attributions (KernelExplainer)...")
    
    pred_res = predictor.predict(input_dict)
    explanation = explainer.explain(input_dict)

    crop_name = explanation["crop_name"]
    confidence_pct = pred_res["confidence_pct"]
    base_val = explanation["base_value"]
    shap_df = explanation["shap_table"]

    print("\n" + "=" * 70)
    print(f"  🎯 PREDICTED CROP : 🌾 {crop_name.upper()} ({confidence_pct}% Confidence)")
    print(f"  📊 BASELINE PROBABILITY : {base_val * 100:.2f}% (Average dataset expectation)")
    print("=" * 70)

    print("\n  🔍 FEATURE ATTRIBUTION BREAKDOWN (SHAP VALUES):")
    print("  " + "─" * 66)
    print(f"  {'Feature Name':24s} | {'Input Val':10s} | {'SHAP Score':11s} | {'Impact'}")
    print("  " + "─" * 66)

    for _, row in shap_df.iterrows():
        fname = row["Feature_Name"]
        uval = f"{row['User_Value']:.1f}"
        sval = row["SHAP_Value"]
        
        # Format direction indicator
        if sval > 1e-4:
            impact_str = f"➕ Positive (+{sval:+.4f})"
        elif sval < -1e-4:
            impact_str = f"➖ Negative ({sval:+.4f})"
        else:
            impact_str = f"⚪ Neutral ({sval:+.4f})"

        print(f"  {fname:24s} | {uval:10s} | {sval:^+11.4f} | {impact_str}")
    print("  " + "─" * 66)

    print("\n  📈 SHAP FEATURE CONTRIBUTION GRAPH:")
    print("  " + "─" * 66)
    for _, row in shap_df.iterrows():
        fname = row["Feature_Name"]
        sval = row["SHAP_Value"]
        
        # Draw ASCII bar graph for positive vs negative contributions
        bar_units = int(abs(sval) * 100)
        if sval > 1e-4:
            bar = "🟩 " + "█" * min(bar_units, 25) + f" (+{sval:.4f})"
        elif sval < -1e-4:
            bar = "🟥 " + "█" * min(bar_units, 25) + f" ({sval:.4f})"
        else:
            bar = "⬜ (0.0000)"
        
        print(f"  {fname:24s} | {bar}")

    print("  " + "─" * 66)

    print("\n  💬 NATURAL LANGUAGE EXPLANATION:")
    print(f"  {explanation['explanation_text']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_terminal_shap_cli()
