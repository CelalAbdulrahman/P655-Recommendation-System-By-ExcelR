from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import Birch, KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import LabelEncoder


def encode_ids(df: pd.DataFrame):
    df = df.copy()

    user_encoder = LabelEncoder()
    product_encoder = LabelEncoder()

    df["user_index"] = user_encoder.fit_transform(df["userId"])
    df["product_index"] = product_encoder.fit_transform(df["productId"])

    return df, user_encoder, product_encoder


def build_sparse_matrix(df: pd.DataFrame):
    matrix = csr_matrix(
        (
            df["rating"].values,
            (df["user_index"].values, df["product_index"].values)
        )
    )
    return matrix


def apply_svd(matrix, n_components: int = 20):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_features = svd.fit_transform(matrix)
    return reduced_features, svd


def evaluate_cluster_model(model_name: str, model, features):
    labels = model.fit_predict(features)

    result = {
        "model_name": model_name,
        "n_clusters": int(len(set(labels))),
        "silhouette_score": float(silhouette_score(features, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(features, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(features, labels))
    }
    return result, labels, model


def compare_clustering_models(features, n_clusters: int = 5):
    results = []

    models = [
        ("KMeans", KMeans(n_clusters=n_clusters, random_state=42, n_init=10)),
        ("MiniBatchKMeans", MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)),
        ("Birch", Birch(n_clusters=n_clusters))
    ]

    fitted_models = {}
    labels_map = {}

    for model_name, model in models:
        result, labels, fitted_model = evaluate_cluster_model(model_name, model, features)
        results.append(result)
        fitted_models[model_name] = fitted_model
        labels_map[model_name] = labels

    results_df = pd.DataFrame(results).sort_values(
        by=["silhouette_score", "calinski_harabasz_score"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return results_df, fitted_models, labels_map


def attach_cluster_labels(df: pd.DataFrame, labels) -> pd.DataFrame:
    user_cluster_df = (
        df[["userId", "user_index"]]
        .drop_duplicates()
        .sort_values("user_index")
        .copy()
    )
    user_cluster_df["cluster"] = labels
    return user_cluster_df


def save_artifacts(svd_model, cluster_model, user_encoder, product_encoder, output_dir: str = "artifacts"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    joblib.dump(svd_model, f"{output_dir}/svd_model.pkl")
    joblib.dump(cluster_model, f"{output_dir}/cluster_model.pkl")
    joblib.dump(user_encoder, f"{output_dir}/user_encoder.pkl")
    joblib.dump(product_encoder, f"{output_dir}/product_encoder.pkl")