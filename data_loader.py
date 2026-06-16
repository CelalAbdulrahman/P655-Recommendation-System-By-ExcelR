import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def basic_info(df: pd.DataFrame) -> dict:
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict()
    }
    return info


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]

    rename_map = {}
    for col in df.columns:
        lower_col = col.lower()

        if lower_col == "userid":
            rename_map[col] = "userId"
        elif lower_col == "productid":
            rename_map[col] = "productId"
        elif lower_col == "rating":
            rename_map[col] = "rating"
        elif lower_col == "timestamp":
            rename_map[col] = "timestamp"
        elif lower_col == "date":
            rename_map[col] = "timestamp"

    df = df.rename(columns=rename_map)
    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    required_cols = ["userId", "productId", "rating"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def prepare_data(file_path: str) -> pd.DataFrame:
    df = load_data(file_path)
    df = standardize_columns(df)
    validate_required_columns(df)
    return df