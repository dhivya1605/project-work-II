import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def preprocess_dataset(file_path):
    """
    Load a synthetic dataset and prepare it for machine learning.
    """

    # Load dataset
    df = pd.read_excel(file_path)

    print("Dataset Loaded Successfully")
    print("Rows :", len(df))
    print("Columns :", len(df.columns))

    # Encode categorical columns
    categorical_columns = [
        "TYPE_OF_CROP",
        "SOIL",
        "SEASON",
        "SOWN",
        "HARVESTED",
        "WATER_SOURCE"
    ]

    encoders = {}

    for column in categorical_columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        encoders[column] = encoder

    # Encode target column
    target_encoder = LabelEncoder()
    df["CROPS"] = target_encoder.fit_transform(df["CROPS"])

    # Features
    X = df.drop("CROPS", axis=1)

    # Target
    y = df["CROPS"]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test