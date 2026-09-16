import pandas as pd
import glob


def find_csv_files(data_dir="data/raw"):
    """
    Find all CSV files inside the raw data directory.
    """

    files = glob.glob(
        f"{data_dir}/**/*.csv",
        recursive=True
    )

    if not files:
        raise FileNotFoundError(
            "No CSV files found in data/raw/. "
            "Please add the Kaggle dataset first."
        )

    return files


def load_dataset(data_dir="data/raw"):
    """
    Load the first CSV file found in the raw data directory.
    """

    files = find_csv_files(data_dir)

    print("CSV files found:")

    for file in files:
        print(f" - {file}")

    print("\nLoading:")
    print(files[0])

    df = pd.read_csv(
        files[0],
        low_memory=False
    )

    print("\nDataset shape:")
    print(df.shape)

    return df


def inspect_dataset(df):
    """
    Print basic information about the dataset.
    """

    print("\n========== DATASET ==========")

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")

    for column in df.columns:
        print(
            f" - {column}"
        )

    print("\nFirst 5 rows:")

    print(
        df.head().to_string()
    )
