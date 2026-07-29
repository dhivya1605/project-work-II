import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. LOAD DATA
# -----------------------------
DATA_PATH = "Synthetic_Crop_Data_Beta_FInal.xlsx"
LABEL_COL = "CROPS"
CATEGORICAL_COLS = ["TYPE_OF_CROP", "SOIL", "SEASON", "WATER_SOURCE"]
DROP_COLS = ["SOWN", "HARVESTED"]

df = pd.read_excel(DATA_PATH)
df.columns = df.columns.str.strip()

# -----------------------------
# 2. PREPROCESS
# -----------------------------
X = df.drop(columns=[LABEL_COL] + DROP_COLS)
X = pd.get_dummies(X, columns=CATEGORICAL_COLS)

le = LabelEncoder()
y = le.fit_transform(df[LABEL_COL])

TRAIN_COLUMNS = X.columns  # save for prediction step later

# -----------------------------
# 3. TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -----------------------------
# 4. TRAIN MODEL
# -----------------------------
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------------
# 5. EVALUATE
# -----------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------------
# 6. PREDICT NEW CROP FROM USER INPUT
# -----------------------------
def predict_new_crop(soil_ph, temp, crop_duration, water_required,
                      humidity, N, P, K,
                      type_of_crop, soil, season, water_source):

    new_data = pd.DataFrame([{
        "SOIL_PH": soil_ph,
        "TEMP": temp,
        "CROPDURATION": crop_duration,
        "WATERREQUIRED": water_required,
        "RELATIVE_HUMIDITY": humidity,
        "N": N, "P": P, "K": K,
        "TYPE_OF_CROP": type_of_crop,
        "SOIL": soil,
        "SEASON": season,
        "WATER_SOURCE": water_source
    }])

    new_data = pd.get_dummies(new_data, columns=CATEGORICAL_COLS)
    new_data = new_data.reindex(columns=TRAIN_COLUMNS, fill_value=0)

    pred = model.predict(new_data)
    return le.inverse_transform(pred)[0]

# Example usage:
result = predict_new_crop(
    soil_ph=6.2, temp=28, crop_duration=100, water_required=600,
    humidity=65, N=80, P=40, K=40,
    type_of_crop="Cereal", soil="Loamy", season="Kharif", water_source="Canal"
)
print("Predicted crop:", result)