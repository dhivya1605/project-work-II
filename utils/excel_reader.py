import pandas as pd

def load_dataset(file_path):
    """
    Reads the Excel dataset.
    """

    df = pd.read_excel(file_path)

    print("Dataset Loaded Successfully")
    print("Number of Crops:", len(df))

    return df