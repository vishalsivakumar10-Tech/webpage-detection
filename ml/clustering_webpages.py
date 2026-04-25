from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_utils import OUTPUT_DIR, load_dataset


def preprocess_features(features: pd.DataFrame):
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
        ]
    )
    transformed = pipeline.fit_transform(features)
    return transformed, pipeline


def evaluate_cluster_counts(transformed_features) -> pd.DataFrame:
    rows = []
    for cluster_count in range(2, 7):
        model = KMeans(n_clusters=cluster_count, n_init=10, random_state=42)
        labels = model.fit_predict(transformed_features)
        rows.append(
            {
                "cluster_count": cluster_count,
                "silhouette_score": silhouette_score(transformed_features, labels),
                "inertia": model.inertia_,
            }
        )
    return pd.DataFrame(rows)


def build_cluster_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        df.groupby("cluster")
        .agg(
            record_count=("Result", "size"),
            phishing_ratio=("Result", lambda series: (series == -1).mean()),
            legitimate_ratio=("Result", lambda series: (series == 1).mean()),
            avg_web_traffic=("web_traffic", "mean"),
            avg_page_rank=("Page_Rank", "mean"),
            avg_links_pointing_to_page=("Links_pointing_to_page", "mean"),
        )
        .reset_index()
    )

    feature_means = (
        df.groupby("cluster")
        .mean(numeric_only=True)
        .drop(columns=["Result"], errors="ignore")
        .reset_index()
    )
    return summary, feature_means


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    features = df.drop(columns=["Result"])
    transformed_features, _ = preprocess_features(features)

    evaluation_df = evaluate_cluster_counts(transformed_features)
    best_cluster_count = int(
        evaluation_df.sort_values(by="silhouette_score", ascending=False).iloc[0]["cluster_count"]
    )

    final_model = KMeans(n_clusters=best_cluster_count, n_init=10, random_state=42)
    df["cluster"] = final_model.fit_predict(transformed_features)

    summary_df, feature_means_df = build_cluster_summary(df)

    evaluation_df.to_csv(OUTPUT_DIR / "clustering_k_selection.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "clustering_summary.csv", index=False)
    feature_means_df.to_csv(OUTPUT_DIR / "clustering_feature_means.csv", index=False)

    print("Clustering Evaluation")
    print(evaluation_df.to_string(index=False))
    print(f"\nSelected cluster count: {best_cluster_count}")
    print("\nCluster Summary")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
