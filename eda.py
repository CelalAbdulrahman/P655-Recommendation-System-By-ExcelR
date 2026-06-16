from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def dataset_summary(df: pd.DataFrame) -> dict:
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "unique_users": int(df["userId"].nunique()),
        "unique_products": int(df["productId"].nunique()),
        "total_ratings": int(df["rating"].count()),
        "rating_min": float(df["rating"].min()),
        "rating_max": float(df["rating"].max()),
        "rating_mean": float(df["rating"].mean())
    }
    return summary


def user_activity_stats(df: pd.DataFrame) -> dict:
    user_counts = df.groupby("userId")["rating"].count()

    stats = {
        "min": int(user_counts.min()),
        "max": int(user_counts.max()),
        "mean": float(user_counts.mean()),
        "median": float(user_counts.median()),
        "users_with_2_or_more_ratings": int((user_counts >= 2).sum())
    }
    return stats


def product_popularity_stats(df: pd.DataFrame) -> dict:
    product_counts = df.groupby("productId")["rating"].count()

    stats = {
        "min": int(product_counts.min()),
        "max": int(product_counts.max()),
        "mean": float(product_counts.mean()),
        "median": float(product_counts.median())
    }
    return stats


def sparsity_info(df: pd.DataFrame) -> dict:
    total_users = df["userId"].nunique()
    total_products = df["productId"].nunique()
    total_possible = total_users * total_products
    observed = len(df)

    sparsity = 1 - (observed / total_possible) if total_possible > 0 else 0

    info = {
        "unique_users": int(total_users),
        "unique_products": int(total_products),
        "observed_interactions": int(observed),
        "possible_interactions": int(total_possible),
        "sparsity": float(sparsity)
    }
    return info


def top_products(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    result = (
        df.groupby("productId")
        .agg(
            rating_count=("rating", "count"),
            avg_rating=("rating", "mean")
        )
        .sort_values(by="rating_count", ascending=False)
        .head(top_n)
        .reset_index()
    )
    return result


def top_users(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    result = (
        df.groupby("userId")
        .agg(
            rating_count=("rating", "count"),
            avg_rating=("rating", "mean")
        )
        .sort_values(by="rating_count", ascending=False)
        .head(top_n)
        .reset_index()
    )
    return result


def save_rating_distribution_plot(df: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(df["rating"], bins=10, edgecolor="black")
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_top_products_plot(df: pd.DataFrame, output_path: str, top_n: int = 10) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    top_df = top_products(df, top_n=top_n)

    plt.figure(figsize=(10, 5))
    plt.bar(top_df["productId"].astype(str), top_df["rating_count"])
    plt.title(f"Top {top_n} Products by Rating Count")
    plt.xlabel("Product ID")
    plt.ylabel("Rating Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_top_users_plot(df: pd.DataFrame, output_path: str, top_n: int = 10) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    top_df = top_users(df, top_n=top_n)

    plt.figure(figsize=(10, 5))
    plt.bar(top_df["userId"].astype(str), top_df["rating_count"])
    plt.title(f"Top {top_n} Users by Rating Count")
    plt.xlabel("User ID")
    plt.ylabel("Rating Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()