from pathlib import Path

import streamlit as st

from src.data_loader import prepare_data
from src.preprocess import clean_data
from src.eda import (
    dataset_summary,
    user_activity_stats,
    product_popularity_stats,
    sparsity_info
)
from src.model_train import (
    encode_ids,
    build_sparse_matrix,
    apply_svd,
    compare_clustering_models,
    attach_cluster_labels
)
from src.recommender import recommend_products, recommend_popular_products


DATA_PATH = "data/rating_short.csv"


@st.cache_data
def load_and_clean_data():
    df = prepare_data(DATA_PATH)
    df = clean_data(df)
    return df


@st.cache_resource
def build_recommendation_system():
    df = load_and_clean_data()

    df_encoded, user_encoder, product_encoder = encode_ids(df)
    matrix = build_sparse_matrix(df_encoded)
    features, svd_model = apply_svd(matrix, n_components=20)

    results_df, fitted_models, labels_map = compare_clustering_models(features, n_clusters=5)

    best_model_name = results_df.iloc[0]["model_name"]
    best_labels = labels_map[best_model_name]

    user_cluster_df = attach_cluster_labels(df_encoded, best_labels)

    return df, user_cluster_df, results_df, best_model_name


def show_saved_plot(plot_path: str, caption: str):
    if Path(plot_path).exists():
        st.image(plot_path, caption=caption, use_container_width=True)


def main():
    st.set_page_config(page_title="Product Recommendation System", layout="wide")

    st.title("Product Recommendation System")
    st.write("Cluster based product recommendation using TruncatedSVD and clustering models.")

    df, user_cluster_df, results_df, best_model_name = build_recommendation_system()

    summary = dataset_summary(df)
    user_stats = user_activity_stats(df)
    product_stats = product_popularity_stats(df)
    sparse_stats = sparsity_info(df)

    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", summary["rows"])
    col2.metric("Users", summary["unique_users"])
    col3.metric("Products", summary["unique_products"])
    col4.metric("Total Ratings", summary["total_ratings"])

    st.subheader("Key Statistics")
    stats_col1, stats_col2 = st.columns(2)

    with stats_col1:
        st.write("User Activity Stats")
        st.json(user_stats)

        st.write("Product Popularity Stats")
        st.json(product_stats)

    with stats_col2:
        st.write("Sparsity Info")
        st.json(sparse_stats)

        st.write("Best Model Selected")
        st.success(best_model_name)

    st.subheader("Clustering Model Comparison")
    st.dataframe(results_df, use_container_width=True)

    st.subheader("Saved EDA Plots")
    plot_col1, plot_col2, plot_col3 = st.columns(3)

    with plot_col1:
        show_saved_plot("plots/rating_distribution.png", "Rating Distribution")

    with plot_col2:
        show_saved_plot("plots/top_products.png", "Top Products")

    with plot_col3:
        show_saved_plot("plots/top_users.png", "Top Users")

    st.subheader("Generate Recommendations")

    default_user = str(df["userId"].iloc[0])
    user_id = st.text_input("Enter User ID", value=default_user)
    top_n = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)

    if st.button("Recommend Products"):
        if not user_id.strip():
            st.warning("Please enter a valid User ID.")
            return

        if user_id in df["userId"].values:
            recommendations = recommend_products(
                df=df,
                user_cluster_df=user_cluster_df,
                user_id=user_id,
                top_n=top_n
            )
            st.success(f"Recommendations generated for user: {user_id}")
        else:
            recommendations = recommend_popular_products(
                df=df,
                user_id=None,
                top_n=top_n
            )
            st.info("User not found. Showing popular product recommendations instead.")

        if recommendations.empty:
            st.warning("No recommendations found.")
        else:
            st.dataframe(recommendations, use_container_width=True)


if __name__ == "__main__":
    main()