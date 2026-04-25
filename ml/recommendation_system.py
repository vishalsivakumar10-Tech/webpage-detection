from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_utils import OUTPUT_DIR, load_dataset


def build_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
        ]
    )


def label_to_text(value: int) -> str:
    return "legitimate" if value == 1 else "phishing"


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend similar webpages from the phishing dataset.")
    parser.add_argument("--query-index", type=int, default=0, help="Row index to use as the query webpage.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations to return.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_dataset().reset_index(drop=True)
    features = df.drop(columns=["Result"])

    preprocessor = build_preprocessor()
    transformed_features = preprocessor.fit_transform(features)

    neighbor_count = min(len(df), max(args.top_k + 5, 10))
    neighbor_model = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine")
    neighbor_model.fit(transformed_features)

    distances, indices = neighbor_model.kneighbors(transformed_features)
    neighbor_labels = df["Result"].to_numpy()[indices[:, 1:]]
    same_class_ratio = (neighbor_labels == df["Result"].to_numpy().reshape(-1, 1)).mean()

    query_index = max(0, min(args.query_index, len(df) - 1))
    query_features = transformed_features[query_index].reshape(1, -1)
    query_distances, query_indices = neighbor_model.kneighbors(query_features)

    recommendation_rows = []
    rank = 1
    for neighbor_index, distance in zip(query_indices[0], query_distances[0]):
        if int(neighbor_index) == query_index:
            continue

        recommendation_rows.append(
            {
                "query_index": query_index,
                "query_label": label_to_text(int(df.loc[query_index, "Result"])),
                "recommended_rank": rank,
                "recommended_index": int(neighbor_index),
                "recommended_label": label_to_text(int(df.loc[neighbor_index, "Result"])),
                "similarity_score": 1 - float(distance),
            }
        )
        rank += 1
        if rank > args.top_k:
            break

    recommendations_df = pd.DataFrame(recommendation_rows)
    evaluation_df = pd.DataFrame(
        [
            {"metric": "same_class_neighbor_ratio", "value": same_class_ratio},
            {
                "metric": "query_neighbor_phishing_ratio",
                "value": (recommendations_df["recommended_label"] == "phishing").mean(),
            },
        ]
    )

    recommendations_df.to_csv(OUTPUT_DIR / "recommendations.csv", index=False)
    evaluation_df.to_csv(OUTPUT_DIR / "recommendation_metrics.csv", index=False)

    print("Recommendation Metrics")
    print(evaluation_df.to_string(index=False))
    print("\nRecommendations")
    print(recommendations_df.to_string(index=False))


if __name__ == "__main__":
    main()
