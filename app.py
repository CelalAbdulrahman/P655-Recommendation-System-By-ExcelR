from src.data_loader import prepare_data
from src.preprocess import clean_data
from src.eda import (
    dataset_summary,
    user_activity_stats,
    product_popularity_stats,
    sparsity_info,
    save_rating_distribution_plot,
    save_top_products_plot,
    save_top_users_plot
)
from src.model_train import (
    encode_ids,
    build_sparse_matrix,
    apply_svd,
    compare_clustering_models,
    attach_cluster_labels,
    save_artifacts
)
from src.recommender import recommend_products


DATA_PATH = "data/rating_short.csv"


def main():
    df = prepare_data(DATA_PATH)
    df = clean_data(df)

    summary = dataset_summary(df)
    user_stats = user_activity_stats(df)
    product_stats = product_popularity_stats(df)
    sparse_stats = sparsity_info(df)

    print("\nDATASET SUMMARY")
    print(summary)

    print("\nUSER ACTIVITY STATS")
    print(user_stats)

    print("\nPRODUCT POPULARITY STATS")
    print(product_stats)

    print("\nSPARSITY INFO")
    print(sparse_stats)

    save_rating_distribution_plot(df, "plots/rating_distribution.png")
    save_top_products_plot(df, "plots/top_products.png", top_n=10)
    save_top_users_plot(df, "plots/top_users.png", top_n=10)

    df_encoded, user_encoder, product_encoder = encode_ids(df)
    matrix = build_sparse_matrix(df_encoded)
    features, svd_model = apply_svd(matrix, n_components=20)

    results_df, fitted_models, labels_map = compare_clustering_models(features, n_clusters=5)

    print("\nCLUSTER MODEL COMPARISON")
    print(results_df)

    best_model_name = results_df.iloc[0]["model_name"]
    best_model = fitted_models[best_model_name]
    best_labels = labels_map[best_model_name]

    print("\nBEST MODEL")
    print(best_model_name)

    user_cluster_df = attach_cluster_labels(df_encoded, best_labels)

    save_artifacts(
        svd_model=svd_model,
        cluster_model=best_model,
        user_encoder=user_encoder,
        product_encoder=product_encoder,
        output_dir="artifacts"
    )

    sample_user_id = df["userId"].iloc[0]
    recommendations = recommend_products(
        df=df,
        user_cluster_df=user_cluster_df,
        user_id=sample_user_id,
        top_n=10
    )

    print(f"\nSAMPLE USER: {sample_user_id}")
    print("\nRECOMMENDATIONS")
    print(recommendations)


if __name__ == "__main__":
    main()