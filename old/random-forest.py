import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ==========================
# 1. LOAD DATASET
# ==========================
df = pd.read_csv("synthetic_crop_datasets1_2000.csv")  # change filename if needed

print("Dataset Shape:", df.shape)

# ==========================
# 2. TARGET COLUMN
# ==========================
TARGET = "CROPS"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ==========================
# 3. ENCODE CATEGORICAL COLUMNS
# ==========================
encoders = {}

for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y.astype(str))

# ==========================
# 4. TRAIN TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# 5. TRAIN RANDOM FOREST
# ==========================
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ==========================
# 6. EVALUATE MODEL
# ==========================
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n============================")
print("RANDOM FOREST RESULTS")
print("============================")
print(f"Accuracy = {accuracy * 100:.2f}%")

# ==========================
# 7. FEATURE IMPORTANCE
# ==========================
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance_df)

# ==========================
# 8. PREDICT SAMPLE ROW
# ==========================
sample_df = X.iloc[[0]]

actual_crop = target_encoder.inverse_transform([y[0]])[0]

prediction = rf.predict(sample_df)

predicted_crop = target_encoder.inverse_transform(prediction)[0]

print("\n============================")
print("SAMPLE PREDICTION")
print("============================")
print("Actual Crop    :", actual_crop)
print("Predicted Crop :", predicted_crop)

# ==========================
# 9. TOP 3 RECOMMENDATIONS
# ==========================
probs = rf.predict_proba(sample_df)[0]

top3 = probs.argsort()[-3:][::-1]

print("\nTop 3 Crop Recommendations:")

for idx in top3:
    crop = target_encoder.inverse_transform([idx])[0]
    probability = probs[idx] * 100

    print(f"{crop} : {probability:.2f}%")