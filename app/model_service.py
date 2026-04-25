from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.feature_extraction import extract_features_from_source, summarize_feature_flags
from ml.data_utils import DATA_PATH


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass
class PredictionResult:
    classification_label: str
    classification_probability: float
    web_traffic_prediction: float
    cluster_id: int
    similar_webpages: list[dict[str, Any]]
    extracted_features: dict[str, float]
    findings: list[dict[str, Any]]
    notes: list[str]
    source_meta: dict[str, Any]


def _load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    return df


def _build_scaled_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
        ]
    )


def _build_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32, 16),
                    activation="relu",
                    solver="adam",
                    alpha=0.0005,
                    batch_size=64,
                    learning_rate_init=0.001,
                    max_iter=300,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                ),
            ),
        ]
    )


def _build_regressor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


class WebpageDetectionService:
    def __init__(self) -> None:
        self.df = _load_dataset().reset_index(drop=True)
        self.feature_columns = [column for column in self.df.columns if column != "Result"]
        self.input_columns = [column for column in self.feature_columns if column != "web_traffic"]

        self.classifier = _build_classifier()
        self.classifier.fit(self.df[self.input_columns], self.df["Result"].map({-1: 0, 1: 1}))

        self.regressor = _build_regressor()
        self.regressor.fit(self.df.drop(columns=["web_traffic"]), self.df["web_traffic"])

        scaled_features = self.df[self.feature_columns]
        self.cluster_preprocessor = _build_scaled_preprocessor()
        transformed_cluster_features = self.cluster_preprocessor.fit_transform(scaled_features)
        self.cluster_model = KMeans(n_clusters=3, n_init=10, random_state=42)
        self.cluster_labels = self.cluster_model.fit_predict(transformed_cluster_features)

        self.recommendation_preprocessor = _build_scaled_preprocessor()
        transformed_recommendation_features = self.recommendation_preprocessor.fit_transform(
            self.df[self.input_columns]
        )
        self.recommendation_model = NearestNeighbors(n_neighbors=6, metric="cosine")
        self.recommendation_model.fit(transformed_recommendation_features)
        self.recommendation_vectors = transformed_recommendation_features

    def get_project_summary(self) -> dict[str, Any]:
        return {
            "dataset_records": int(len(self.df)),
            "input_features": len(self.input_columns),
            "classification_metrics": self._load_metric_file("neural_network_metrics.csv"),
            "regression_metrics": self._load_metric_file("regression_metrics.csv"),
            "clusters": self._load_cluster_summary(),
        }

    def predict(self, payload: dict[str, float]) -> PredictionResult:
        row = self._build_feature_row(payload)
        return self._predict_from_row(row, notes=[], source_meta={})

    def analyze_source(
        self,
        url: str = "",
        text: str = "",
        html: str = "",
        overrides: dict[str, float] | None = None,
    ) -> PredictionResult:
        extracted = extract_features_from_source(url=url, text=text, html=html)
        feature_payload = extracted.features.copy()
        if overrides:
            for key, value in overrides.items():
                if key in self.input_columns and value is not None:
                    feature_payload[key] = float(value)

        row = self._build_feature_row(feature_payload)
        notes = list(extracted.notes)
        if url:
            notes.insert(0, f"Analyzed source URL: {url.strip()}")
        if extracted.fetched:
            notes.append("Live page content was fetched automatically for feature extraction.")
        elif url and not text and not html:
            notes.append("Analysis used URL structure only because live content could not be fetched.")

        source_meta = {
            "url": url.strip(),
            "fetched_live_content": extracted.fetched,
            "page_text_length": extracted.page_text_length,
        }
        return self._predict_from_row(row, notes=notes, source_meta=source_meta)

    def _predict_from_row(
        self,
        row: pd.DataFrame,
        notes: list[str],
        source_meta: dict[str, Any],
    ) -> PredictionResult:

        classification_prob = float(self.classifier.predict_proba(row[self.input_columns])[0][1])
        classification_label = "Legitimate" if classification_prob >= 0.5 else "Phishing"

        regression_input = row.copy()
        regression_input["Result"] = 1 if classification_prob >= 0.5 else -1
        predicted_traffic = float(self.regressor.predict(regression_input)[0])
        regression_input["web_traffic"] = predicted_traffic

        transformed_cluster_row = self.cluster_preprocessor.transform(regression_input[self.feature_columns])
        cluster_id = int(self.cluster_model.predict(transformed_cluster_row)[0])

        similar_webpages = self._recommend_similar(row[self.input_columns], classification_label)
        findings = summarize_feature_flags(row.iloc[0].to_dict())

        return PredictionResult(
            classification_label=classification_label,
            classification_probability=classification_prob,
            web_traffic_prediction=predicted_traffic,
            cluster_id=cluster_id,
            similar_webpages=similar_webpages,
            extracted_features={column: float(row.iloc[0][column]) for column in self.input_columns},
            findings=findings,
            notes=notes,
            source_meta=source_meta,
        )

    def _build_feature_row(self, payload: dict[str, float]) -> pd.DataFrame:
        values = {}
        dataset_means = self.df[self.input_columns].mean(numeric_only=True)
        for column in self.input_columns:
            value = payload.get(column)
            values[column] = float(value) if value is not None else float(dataset_means[column])
        return pd.DataFrame([values], columns=self.input_columns)

    def _recommend_similar(
        self, input_frame: pd.DataFrame, predicted_label: str, top_k: int = 4
    ) -> list[dict[str, Any]]:
        transformed = self.recommendation_preprocessor.transform(input_frame)
        distances, indices = self.recommendation_model.kneighbors(transformed)

        rows: list[dict[str, Any]] = []
        for rank, (neighbor_index, distance) in enumerate(zip(indices[0], distances[0]), start=1):
            neighbor = self.df.iloc[int(neighbor_index)]
            rows.append(
                {
                    "rank": rank,
                    "dataset_index": int(neighbor_index),
                    "label": "Legitimate" if int(neighbor["Result"]) == 1 else "Phishing",
                    "similarity": round(1 - float(distance), 4),
                    "web_traffic": float(neighbor["web_traffic"]),
                    "match_signal": "Same predicted class"
                    if (
                        predicted_label == "Legitimate" and int(neighbor["Result"]) == 1
                    )
                    or (predicted_label == "Phishing" and int(neighbor["Result"]) == -1)
                    else "Opposite class",
                }
            )
            if len(rows) >= top_k:
                break
        return rows

    def _load_metric_file(self, filename: str) -> dict[str, float]:
        path = OUTPUT_DIR / filename
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        return {str(row["metric"]): float(row["value"]) for _, row in df.iterrows()}

    def _load_cluster_summary(self) -> list[dict[str, Any]]:
        path = OUTPUT_DIR / "clustering_summary.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path)
        return df.to_dict(orient="records")


service = WebpageDetectionService()
