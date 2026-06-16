import pandas as pd


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    return df


def remove_missing_required(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["userId", "productId", "rating"]).copy()
    return df


def convert_rating_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"]).copy()
    return df


def keep_users_with_min_ratings(df: pd.DataFrame, min_ratings: int = 2) -> pd.DataFrame:
    user_counts = df["userId"].value_counts()
    valid_users = user_counts[user_counts >= min_ratings].index
    df = df[df["userId"].isin(valid_users)].copy()
    return df


def keep_products_with_min_ratings(df: pd.DataFrame, min_ratings: int = 1) -> pd.DataFrame:
    product_counts = df["productId"].value_counts()
    valid_products = product_counts[product_counts >= min_ratings].index
    df = df[df["productId"].isin(valid_products)].copy()
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_duplicate_rows(df)
    df = remove_missing_required(df)
    df = convert_rating_type(df)
    df = keep_users_with_min_ratings(df, min_ratings=2)
    df = keep_products_with_min_ratings(df, min_ratings=1)
    return df