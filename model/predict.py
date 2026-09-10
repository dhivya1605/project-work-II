"""
predict.py
----------
Inference module for the trained TabKANet Crop Recommendation model.

This module reuses the EXACT trained model weights, feature selector configuration,
scaler, and label encoder saved in `model/artifacts/`. It performs zero retraining
or modification of the trained weights.

Features expected by the full dataset:
  - SOIL_PH
  - N
  - P
  - K
  - TEMP
  - WATERREQUIRED
  - RELATIVE_HUMIDITY
  - CROPDURATION

SFOA-selected features (loaded dynamically from model_config.json):
  - TEMP, WATERREQUIRED, RELATIVE_HUMIDITY, CROPDURATION
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tabkanet import TabKANet

DEFAULT_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

# Default agronomic fallback values for optional features if omitted
DEFAULT_FEATURE_VALUES = {
    "N": 50.0,
    "P": 40.0,
    "K": 40.0,
    "SOIL_PH": 6.5,
    "TEMP": 25.0,
    "WATERREQUIRED": 1000.0,
    "RELATIVE_HUMIDITY": 70.0,
    "CROPDURATION": 120.0,
}

# Standard display names mapping
FEATURE_DISPLAY_NAMES = {
    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "SOIL_PH": "Soil pH",
    "TEMP": "Temperature (°C)",
    "WATERREQUIRED": "Rainfall / Water (mm)",
    "RELATIVE_HUMIDITY": "Relative Humidity (%)",
    "CROPDURATION": "Crop Duration (Days)",
}


class CropPredictor:
    def __init__(self, artifacts_dir: str = DEFAULT_ARTIFACTS_DIR):
        self.artifacts_dir = os.path.abspath(artifacts_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.config = None
        self.scaler = None
        self.label_encoder = None
        self.model = None
        self.selected_columns = []
        
        self._load_artifacts()

    def _load_artifacts(self):
        config_path = os.path.join(self.artifacts_dir, "model_config.json")
        scaler_path = os.path.join(self.artifacts_dir, "scaler.pkl")
        encoder_path = os.path.join(self.artifacts_dir, "label_encoder.pkl")
        weights_path = os.path.join(self.artifacts_dir, "tabkanet_weights.pt")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder file not found at {encoder_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights file not found at {weights_path}")

        # 1. Load config
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.selected_columns = self.config.get("selected_columns", [])
        num_features = self.config.get("num_features", len(self.selected_columns))
        num_classes = self.config.get("num_classes", 49)
        hyperparams = self.config.get("hyperparameters", {})

        # 2. Load sklearn artifacts
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)

        # 3. Instantiate TabKANet model architecture matching saved hyperparameters
        self.model = TabKANet(
            num_features=num_features,
            num_classes=num_classes,
            embed_dim=hyperparams.get("embed_dim", 32),
            n_heads=hyperparams.get("n_heads", 2),
            n_transformer_layers=hyperparams.get("n_transformer_layers", 1),
            ff_dim=hyperparams.get("ff_dim", 96),
            dropout=hyperparams.get("dropout", 0.0),
        ).to(self.device)

        # 4. Load trained weights
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def prepare_input_dataframe(self, input_dict: dict) -> pd.DataFrame:
        """
        Converts input dictionary (or row) to a standardized pandas DataFrame
        with proper column names.
        """
        row = {}
        # Normalize column key names
        normalized_input = {k.upper().strip(): v for k, v in input_dict.items()}
        
        # Map common aliases
        aliases = {
            "RAINFALL": "WATERREQUIRED",
            "WATER_REQUIRED": "WATERREQUIRED",
            "WATER": "WATERREQUIRED",
            "TEMPERATURE": "TEMP",
            "PH": "SOIL_PH",
            "SOILPH": "SOIL_PH",
            "HUMIDITY": "RELATIVE_HUMIDITY",
            "DURATION": "CROPDURATION",
        }
        for alias, target in aliases.items():
            if alias in normalized_input and target not in normalized_input:
                normalized_input[target] = normalized_input[alias]

        # Fill all 8 feature columns
        all_cols = ["SOIL_PH", "N", "P", "K", "TEMP", "WATERREQUIRED", "RELATIVE_HUMIDITY", "CROPDURATION"]
        for col in all_cols:
            if col in normalized_input and normalized_input[col] is not None:
                row[col] = float(normalized_input[col])
            else:
                row[col] = DEFAULT_FEATURE_VALUES.get(col, 0.0)

        return pd.DataFrame([row])

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extracts SFOA-selected columns in order and applies StandardScaler.
        """
        # Ensure all selected columns are present
        missing = [col for col in self.selected_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns for model inference: {missing}")

        X_sel = df[self.selected_columns].values.astype(np.float32)
        X_scaled = self.scaler.transform(X_sel).astype(np.float32)
        return X_scaled

    def predict(self, input_data) -> dict:
        """
        Runs model inference on input_data (dict, pandas DataFrame, or 2D array).

        Returns:
            dict containing:
                - 'predicted_crop': string (crop name)
                - 'predicted_class_index': int
                - 'confidence': float (0.0 to 1.0)
                - 'confidence_pct': float (0.0 to 100.0)
                - 'all_probabilities': dict mapping crop_name -> float prob
                - 'top_k_crops': list of tuples (crop_name, prob)
                - 'input_df': pd.DataFrame of processed input features
        """
        if isinstance(input_data, dict):
            df = self.prepare_input_dataframe(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        elif isinstance(input_data, np.ndarray):
            # Assume array corresponds to selected_columns if 4 dims, or full 8 cols if 8 dims
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)
            if input_data.shape[1] == len(self.selected_columns):
                df = pd.DataFrame(input_data, columns=self.selected_columns)
            else:
                all_cols = ["SOIL_PH", "N", "P", "K", "TEMP", "WATERREQUIRED", "RELATIVE_HUMIDITY", "CROPDURATION"]
                df = pd.DataFrame(input_data, columns=all_cols[:input_data.shape[1]])
        else:
            raise TypeError(f"Unsupported input_data type: {type(input_data)}")

        X_scaled = self.preprocess(df)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        crop_name = str(self.label_encoder.inverse_transform([pred_idx])[0])

        all_classes = self.label_encoder.classes_
        prob_dict = {str(all_classes[i]): float(probs[i]) for i in range(len(all_classes))}
        
        # Sort top-5 predictions
        top_k_indices = np.argsort(probs)[::-1][:5]
        top_k_crops = [(str(all_classes[i]), float(probs[i])) for i in top_k_indices]

        return {
            "predicted_crop": crop_name,
            "predicted_class_index": pred_idx,
            "confidence": confidence,
            "confidence_pct": round(confidence * 100, 2),
            "all_probabilities": prob_dict,
            "top_k_crops": top_k_crops,
            "input_df": df,
            "scaled_features": X_scaled,
        }


def run_interactive_cli():
    print("=" * 65)
    print("  🌾 TABKANET EXPLAINABLE CROP RECOMMENDATION SYSTEM (CLI)")
    print("=" * 65)
    print("Loading trained model artifacts from ./artifacts/ ...")
    
    predictor = CropPredictor()
    print("✅ Model loaded successfully!")
    print("\nPlease enter the agronomic & environmental values below.")
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

    print("\n" + "─" * 65)
    print("Running TabKANet inference...")
    result = predictor.predict(input_dict)

    print("\n" + "=" * 65)
    print("  🎯 PREDICTION RESULT")
    print("=" * 65)
    print(f"  Recommended Crop : 🌾 {result['predicted_crop'].upper()}")
    print(f"  Confidence Score : 📊 {result['confidence_pct']}%")
    print("─" * 65)
    print("  Top 5 Predictions Probability Distribution:")
    for rank, (crop, prob) in enumerate(result['top_k_crops'], 1):
        bar_len = int(prob * 25)
        bar = "█" * bar_len
        print(f"    {rank}. {crop.title():18s} | {prob * 100:6.2f}%  {bar}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_interactive_cli()
