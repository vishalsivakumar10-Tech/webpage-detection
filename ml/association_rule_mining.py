from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "dataset" / "uci-ml-phishing-dataset (1).csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
TARGET_VALUES = ("Result=-1", "Result=1")


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    return df


def to_transactions(frame: pd.DataFrame) -> list[set[str]]:
    transactions: list[set[str]] = []
    for _, row in frame.iterrows():
        transactions.append({f"{column}={row[column]}" for column in frame.columns})
    return transactions


def generate_frequent_1_itemsets(
    transactions: list[set[str]], min_support: float
) -> dict[frozenset[str], float]:
    counter: Counter[str] = Counter()
    for transaction in transactions:
        for item in transaction:
            counter[item] += 1

    transaction_count = len(transactions)
    return {
        frozenset([item]): count / transaction_count
        for item, count in counter.items()
        if (count / transaction_count) >= min_support
    }


def generate_frequent_2_itemsets(
    transactions: list[set[str]],
    frequent_1_itemsets: dict[frozenset[str], float],
    min_support: float,
) -> dict[frozenset[str], float]:
    pairs: Counter[frozenset[str]] = Counter()
    keys = list(frequent_1_itemsets.keys())
    transaction_count = len(transactions)

    for transaction in transactions:
        present_items = [itemset for itemset in keys if itemset.issubset(transaction)]
        for left, right in combinations(present_items, 2):
            pairs[left | right] += 1

    return {
        itemset: count / transaction_count
        for itemset, count in pairs.items()
        if (count / transaction_count) >= min_support
    }


def build_rules(
    frequent_1_itemsets: dict[frozenset[str], float],
    frequent_2_itemsets: dict[frozenset[str], float],
    min_confidence: float,
) -> pd.DataFrame:
    rules: list[dict[str, float | str]] = []

    for itemset, support in frequent_2_itemsets.items():
        for rhs_label in TARGET_VALUES:
            rhs = frozenset([rhs_label])
            if not rhs.issubset(itemset):
                continue

            lhs = itemset - rhs
            lhs_support = frequent_1_itemsets.get(lhs)
            rhs_support = frequent_1_itemsets.get(rhs)

            if not lhs_support or not rhs_support:
                continue

            confidence = support / lhs_support
            if confidence < min_confidence:
                continue

            lift = confidence / rhs_support
            leverage = support - (lhs_support * rhs_support)
            conviction = (1 - rhs_support) / (1 - confidence) if confidence < 1 else float("inf")

            rules.append(
                {
                    "antecedent": next(iter(lhs)),
                    "consequent": rhs_label,
                    "support": support,
                    "confidence": confidence,
                    "lift": lift,
                    "leverage": leverage,
                    "conviction": conviction,
                }
            )

    if not rules:
        return pd.DataFrame(
            columns=[
                "antecedent",
                "consequent",
                "support",
                "confidence",
                "lift",
                "leverage",
                "conviction",
            ]
        )

    return pd.DataFrame(rules).sort_values(
        by=["lift", "confidence", "support"],
        ascending=False,
    )


def evaluate_rules(rules_df: pd.DataFrame, test_transactions: list[set[str]]) -> pd.DataFrame:
    if rules_df.empty:
        return rules_df

    evaluated_rows = []
    test_size = len(test_transactions)

    for row in rules_df.to_dict(orient="records"):
        antecedent = row["antecedent"]
        consequent = row["consequent"]

        covered = 0
        correct = 0
        for transaction in test_transactions:
            if antecedent in transaction:
                covered += 1
                if consequent in transaction:
                    correct += 1

        row["test_coverage"] = covered / test_size if test_size else 0
        row["test_confidence"] = correct / covered if covered else 0
        evaluated_rows.append(row)

    return pd.DataFrame(evaluated_rows)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Result"],
    )

    train_transactions = to_transactions(train_df)
    test_transactions = to_transactions(test_df)

    min_support = 0.20
    min_confidence = 0.80

    frequent_1_itemsets = generate_frequent_1_itemsets(train_transactions, min_support)
    frequent_2_itemsets = generate_frequent_2_itemsets(
        train_transactions,
        frequent_1_itemsets,
        min_support,
    )

    frequent_itemsets_df = pd.DataFrame(
        [
            {"itemset": ", ".join(sorted(itemset)), "support": support, "size": len(itemset)}
            for itemset, support in {**frequent_1_itemsets, **frequent_2_itemsets}.items()
        ]
    ).sort_values(by=["size", "support"], ascending=[True, False])

    rules_df = build_rules(frequent_1_itemsets, frequent_2_itemsets, min_confidence)
    evaluated_rules_df = evaluate_rules(rules_df, test_transactions)

    frequent_itemsets_df.to_csv(OUTPUT_DIR / "frequent_itemsets.csv", index=False)
    evaluated_rules_df.to_csv(OUTPUT_DIR / "association_rules.csv", index=False)

    print("Frequent Itemsets")
    print(frequent_itemsets_df.head(15).to_string(index=False))
    print("\nAssociation Rules")
    print(evaluated_rules_df.to_string(index=False))


if __name__ == "__main__":
    main()
