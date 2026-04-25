from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "dataset" / "uci-ml-phishing-dataset (1).csv"
OUTPUT_DIR = ROOT_DIR / "outputs"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    return df


def load_feature_frame() -> pd.DataFrame:
    df = load_dataset()
    return df.drop(columns=["Result"])


def load_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    df = load_dataset()
    features = df.drop(columns=["Result"])
    target = df["Result"].map({-1: 0, 1: 1})
    return features, target
