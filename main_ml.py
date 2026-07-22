from preprocessing.preprocess import preprocess_dataset

from models.random_forest import train_random_forest
from models.xgboost import train_xgboost

# Choose dataset
dataset = "output/Synthetic_Crop_Data_Beta_Final.xlsx"

# Preprocess
X_train, X_test, y_train, y_test = preprocess_dataset(dataset)

print("\nTraining Random Forest...\n")

rf_results = train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test
)

print(rf_results)

print("\nTraining XGBoost...\n")

xgb_results = train_xgboost(
    X_train,
    X_test,
    y_train,
    y_test
)

print(xgb_results)