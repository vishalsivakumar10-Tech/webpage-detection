from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_utils import OUTPUT_DIR, load_dataset


TARGET_COLUMN = "web_traffic"


def build_model() -> Pipeline:
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


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    model = build_model()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    holdout_metrics = pd.DataFrame(
        [
            {"metric": "mae", "value": mean_absolute_error(y_test, predictions)},
            {"metric": "rmse", "value": mean_squared_error(y_test, predictions) ** 0.5},
            {"metric": "r2", "value": r2_score(y_test, predictions)},
        ]
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    scores = cross_validate(model, features, target, cv=cv, scoring=scoring, n_jobs=1)
    cv_metrics = pd.DataFrame(
        [
            {
                "metric": "mae",
                "mean": -scores["test_mae"].mean(),
                "std": scores["test_mae"].std(),
            },
            {
                "metric": "rmse",
                "mean": -scores["test_rmse"].mean(),
                "std": scores["test_rmse"].std(),
            },
            {
                "metric": "r2",
                "mean": scores["test_r2"].mean(),
                "std": scores["test_r2"].std(),
            },
        ]
    )

    feature_importance_df = pd.DataFrame(
        {
            "feature": features.columns,
            "importance": model.named_steps["regressor"].feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    sample_predictions_df = pd.DataFrame(
        {
            "actual_web_traffic": y_test.reset_index(drop=True),
            "predicted_web_traffic": predictions,
        }
    ).head(25)

    holdout_metrics.to_csv(OUTPUT_DIR / "regression_metrics.csv", index=False)
    cv_metrics.to_csv(OUTPUT_DIR / "regression_cv_metrics.csv", index=False)
    feature_importance_df.to_csv(OUTPUT_DIR / "regression_feature_importance.csv", index=False)
    sample_predictions_df.to_csv(OUTPUT_DIR / "regression_sample_predictions.csv", index=False)

    print("Regression Metrics")
    print(holdout_metrics.to_string(index=False))
    print("\nCross-Validation Metrics")
    print(cv_metrics.to_string(index=False))
    print("\nTop Features")
    print(feature_importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
