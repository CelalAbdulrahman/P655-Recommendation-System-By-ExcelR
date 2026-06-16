import pandas as pd


def build_product_scores(df: pd.DataFrame) -> pd.DataFrame:
    product_stats = (
        df.groupby("productId")
        .agg(
            rating_count=("rating", "count"),
            avg_rating=("rating", "mean")
        )
        .reset_index()
    )

    global_mean = df["rating"].mean()
    m = product_stats["rating_count"].quantile(0.75)

    product_stats["weighted_score"] = (
        (product_stats["rating_count"] / (product_stats["rating_count"] + m)) * product_stats["avg_rating"]
        + (m / (product_stats["rating_count"] + m)) * global_mean
    )

    product_stats = product_stats.sort_values(
        by=["weighted_score", "rating_count"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return product_stats


def get_user_seen_products(df: pd.DataFrame, user_id: str) -> set:
    seen_products = set(df.loc[df["userId"] == user_id, "productId"].tolist())
    return seen_products


def get_user_cluster(user_cluster_df: pd.DataFrame, user_id: str):
    row = user_cluster_df.loc[user_cluster_df["userId"] == user_id]

    if row.empty:
        return None

    return int(row["cluster"].iloc[0])


def recommend_from_cluster(
    df: pd.DataFrame,
    user_cluster_df: pd.DataFrame,
    user_id: str,
    top_n: int = 10
) -> pd.DataFrame:
    user_cluster = get_user_cluster(user_cluster_df, user_id)

    if user_cluster is None:
        return pd.DataFrame()

    cluster_users = user_cluster_df.loc[
        user_cluster_df["cluster"] == user_cluster, "userId"
    ].tolist()

    cluster_df = df[df["userId"].isin(cluster_users)].copy()
    seen_products = get_user_seen_products(df, user_id)

    recommendations = build_product_scores(cluster_df)
    recommendations = recommendations[~recommendations["productId"].isin(seen_products)].copy()

    recommendations.insert(0, "cluster", user_cluster)
    recommendations = recommendations.head(top_n).reset_index(drop=True)

    return recommendations


def recommend_popular_products(
    df: pd.DataFrame,
    user_id: str = None,
    top_n: int = 10
) -> pd.DataFrame:
    recommendations = build_product_scores(df)

    if user_id is not None:
        seen_products = get_user_seen_products(df, user_id)
        recommendations = recommendations[~recommendations["productId"].isin(seen_products)].copy()

    recommendations = recommendations.head(top_n).reset_index(drop=True)
    return recommendations


def recommend_products(
    df: pd.DataFrame,
    user_cluster_df: pd.DataFrame,
    user_id: str,
    top_n: int = 10
) -> pd.DataFrame:
    cluster_recommendations = recommend_from_cluster(
        df=df,
        user_cluster_df=user_cluster_df,
        user_id=user_id,
        top_n=top_n
    )

    if not cluster_recommendations.empty:
        return cluster_recommendations

    fallback_recommendations = recommend_popular_products(
        df=df,
        user_id=user_id,
        top_n=top_n
    )

    if "cluster" not in fallback_recommendations.columns:
        fallback_recommendations.insert(0, "cluster", "Not Available")

    return fallback_recommendations