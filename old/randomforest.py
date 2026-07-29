import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==============================
# Config
# ==============================
FILE_PATH = "Synthetic_Crop_Datasets.xlsx"   # Change filename if needed
SHEET_NAME = "LHS"                            # LHS | Sobol | TruncatedNormal | GaussianCopula | Beta | Hybrid

# Columns that must NOT go into X:
#   Sample_ID    -> a row counter laid out in contiguous per-crop blocks
#                   (rows 1-200 = crop 1, 201-400 = crop 2, ...). RF can
#                   trivially split on it and "predict" the crop from its
#                   own row position -> this is what caused your 100%/100%
#                   result, not real signal.
#   Technique    -> constant string within a sheet, not a real feature.
#   Drought_Risk -> empty in the source table (all NaN / becomes the
#                   literal string "nan" after your object-column encoding
#                   loop), carries zero information either way.
DROP_COLS = ["Sample_ID", "Technique", "Drought_Risk"]

# ==============================
# Load Dataset
# ==============================
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)   # sheet_name added: workbook has 6 sheets

# Drop leakage / non-feature columns BEFORE encoding, so they can never
# leak back in even if someone changes the encoding loop below later.
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# ==============================
# Encode Categorical Columns
# ==============================
label_encoders = {}

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ==============================
# Features and Target
# ==============================
X = df.drop("CROPS", axis=1)
y = df["CROPS"]

# ==============================
# Train Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# Random Forest Model
# ==============================
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

# ==============================
# Predictions
# ==============================
train_pred = rf.predict(X_train)
test_pred = rf.predict(X_test)

# ==============================
# Metrics
# ==============================
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("="*50)
print("SHEET             :", SHEET_NAME)
print("TRAINING ACCURACY :", round(train_acc*100,2),"%")
print("TEST ACCURACY     :", round(test_acc*100,2),"%")
print("="*50)

print("\nPrecision :", precision_score(y_test,test_pred,average='weighted'))
print("Recall    :", recall_score(y_test,test_pred,average='weighted'))
print("F1 Score  :", f1_score(y_test,test_pred,average='weighted'))

print("\nClassification Report")
print(classification_report(y_test,test_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test,test_pred))

# ==============================
# Feature Importance
# ==============================
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance")
print(importance)

# ==============================
# Overfitting Check
# ==============================
print("\n" + "="*50)

difference = train_acc - test_acc

if difference < 0.02:
    print("Model is NOT overfitting.")
elif difference < 0.05:
    print("Slight overfitting.")
else:
    print("Overfitting detected.")

print("Train-Test Accuracy Difference :", round(difference*100,2),"%")
print("="*50)