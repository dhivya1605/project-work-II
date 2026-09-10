"""
run_terminal_demo.py
--------------------
Master Terminal Suite for Project Review.
Provides terminal execution for:
  1. Standalone KAN vs TabKANet Accuracy Comparison & Interpretation
  2. Full SFOA-DLO-TabKANet Pipeline Training with Metrics (Accuracy, Precision, Recall, F1)
  3. Interactive Crop Prediction & SHAP Local Feature Attribution in Terminal
"""

import sys
import os

from compare_models import run_comparison
from train import main as run_training_pipeline
from shap_explainer import run_terminal_shap_cli

def main_menu():
    while True:
        print("\n" + "=" * 70)
        print("  🌾 EXPLAINABLE TABKANET CROP RECOMMENDATION SYSTEM - TERMINAL REVIEW")
        print("=" * 70)
        print("  1. 📊 Compare Accuracy: Standalone KAN Layer vs. TabKANet Model")
        print("  2. ⚙️  Run Full SFOA-DLO-TabKANet Training Pipeline (Metrics: Acc, Prec, Rec, F1)")
        print("  3. 🧠 Interactive Crop Prediction & SHAP Feature Attribution Explanation (Terminal)")
        print("  4. ❌ Exit")
        print("=" * 70)

        choice = input("  Select an option [1-4]: ").strip()

        if choice == "1":
            run_comparison()
        elif choice == "2":
            run_training_pipeline()
        elif choice == "3":
            run_terminal_shap_cli()
        elif choice == "4" or choice.lower() == "exit":
            print("\nExiting Terminal Review Suite. Good luck with your review! 🚀\n")
            sys.exit(0)
        else:
            print("⚠️ Invalid choice. Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    main_menu()
