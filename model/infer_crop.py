"""
infer_crop.py
-------------
Interactive and programmatic script to take agronomic inputs (N, P, K, pH, Temp, etc.)
and predict the exact crop using the trained TabKANet model artifacts.
"""

import sys
from predict import CropPredictor

def run_inference(sample_inputs: dict):
    print("\n" + "=" * 60)
    print(" 🌾 TABKANET CROP RECOMMENDATION INFERENCE")
    print("=" * 60)
    
    predictor = CropPredictor()
    result = predictor.predict(sample_inputs)

    print("\n--- Input Parameters ---")
    for k, v in sample_inputs.items():
        print(f"  {k:20s}: {v}")

    print("\n--- Model Prediction Results ---")
    print(f"  🎯 Recommended Crop  : {result['predicted_crop'].upper()}")
    print(f"  📊 Confidence Score  : {result['confidence_pct']}%")

    print("\n--- Top 5 Recommended Crops ---")
    for rank, (crop, prob) in enumerate(result['top_k_crops'], 1):
        bar = "█" * int(prob * 20)
        print(f"  {rank}. {crop:20s} | {prob * 100:6.2f}% {bar}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    # Sample input agronomic conditions
    input_data = {
        "N": 80.0,
        "P": 45.0,
        "K": 40.0,
        "SOIL_PH": 6.5,
        "TEMP": 27.5,
        "WATERREQUIRED": 1100.0,
        "RELATIVE_HUMIDITY": 78.0,
        "CROPDURATION": 110.0,
    }
    
    # If user provided command-line arguments, parse them:
    # Example: python infer_crop.py N=90 P=40 K=40 SOIL_PH=6.2 TEMP=28 WATERREQUIRED=1200 RELATIVE_HUMIDITY=75 CROPDURATION=120
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if "=" in arg:
                k, v = arg.split("=", 1)
                input_data[k.upper().strip()] = float(v)

    run_inference(input_data)
