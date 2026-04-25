from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_utils import OUTPUT_DIR, load_classification_data


def build_model() -> Pipeline:
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


def evaluate_holdout(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    metrics = pd.DataFrame(
        [
            {"metric": "accuracy", "value": accuracy_score(y_test, y_pred)},
            {"metric": "precision", "value": precision_score(y_test, y_pred)},
            {"metric": "recall", "value": recall_score(y_test, y_pred)},
            {"metric": "f1_score", "value": f1_score(y_test, y_pred)},
            {"metric": "roc_auc", "value": roc_auc_score(y_test, y_prob)},
        ]
    )

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["actual_phishing", "actual_legitimate"],
        columns=["predicted_phishing", "predicted_legitimate"],
    )
    return metrics, cm_df


def evaluate_cross_validation(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    model = build_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(
        model,
        features,
        target,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        n_jobs=1,
    )

    rows = []
    for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        values = scores[f"test_{metric_name}"]
        rows.append(
            {
                "metric": metric_name,
                "mean": values.mean(),
                "std": values.std(),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    features, target = load_classification_data()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = build_model()
    model.fit(x_train, y_train)

    holdout_metrics, confusion_df = evaluate_holdout(model, x_test, y_test)
    cv_metrics = evaluate_cross_validation(features, target)

    holdout_metrics.to_csv(OUTPUT_DIR / "neural_network_metrics.csv", index=False)
    confusion_df.to_csv(OUTPUT_DIR / "neural_network_confusion_matrix.csv")
    cv_metrics.to_csv(OUTPUT_DIR / "neural_network_cv_metrics.csv", index=False)

    print("Neural Network Model Performance")
    print(holdout_metrics.to_string(index=False))
    print("\nConfusion Matrix")
    print(confusion_df.to_string())
    print("\n5-Fold Cross-Validation Metrics")
    print(cv_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
