# Apriori Algorithm Mathematical Explanation

## Introduction

Apriori is an Association Rule Mining algorithm used to discover frequent patterns in transactional data. In this project, each website record is converted into a transaction of feature-value items such as:

- `having_IP_Address=-1`
- `SSLfinal_State=-1`
- `URL_of_Anchor=-1`
- `Result=-1`

The goal is to find:

1. Frequent itemsets
2. Strong association rules

## Basic Definitions

Let:

- `I = {i1, i2, i3, ..., in}` be the set of all items
- `D = {T1, T2, T3, ..., Tm}` be the transaction database
- Each transaction `T` is a subset of `I`

An itemset `X` is frequent if its support is greater than or equal to the minimum support threshold.

## Support

Support measures how frequently an itemset appears in the dataset.

For an itemset `X`:

```text
Support(X) = Number of transactions containing X / Total number of transactions
```

Mathematically:

```text
support(X) = |{T in D : X subseteq T}| / |D|
```

Example:

If `URL_of_Anchor=-1` appears in 3200 transactions out of 11055:

```text
support(URL_of_Anchor=-1) = 3200 / 11055
```

## Association Rule

An association rule is written as:

```text
X => Y
```

where:

- `X` is the antecedent
- `Y` is the consequent
- `X ∩ Y = empty set`

Example:

```text
URL_of_Anchor=-1 => Result=-1
```

## Confidence

Confidence measures how often `Y` is true when `X` is true.

```text
confidence(X => Y) = support(X union Y) / support(X)
```

It estimates the conditional probability:

```text
P(Y | X)
```

Example:

```text
confidence(URL_of_Anchor=-1 => Result=-1)
= support(URL_of_Anchor=-1, Result=-1) / support(URL_of_Anchor=-1)
```

## Lift

Lift measures how much stronger the rule is compared to random occurrence.

```text
lift(X => Y) = confidence(X => Y) / support(Y)
```

Equivalent form:

```text
lift(X => Y) = support(X union Y) / (support(X) * support(Y))
```

Interpretation:

- `lift > 1` means positive association
- `lift = 1` means independent
- `lift < 1` means negative association

## Leverage

Leverage measures the difference between observed co-occurrence and expected co-occurrence if `X` and `Y` were independent.

```text
leverage(X => Y) = support(X union Y) - support(X) * support(Y)
```

Higher leverage means a stronger useful association.

## Conviction

Conviction measures the dependence of `Y` on `X`.

```text
conviction(X => Y) = (1 - support(Y)) / (1 - confidence(X => Y))
```

If confidence is very high, conviction also becomes high.

## Apriori Principle

The Apriori algorithm is based on this key property:

```text
If an itemset is frequent, all of its non-empty subsets must also be frequent.
```

This is called the **downward closure property**.

It helps reduce computation because:

- If an itemset is infrequent, all larger itemsets containing it can be pruned.

## Apriori Steps

### Step 1: Generate frequent 1-itemsets

Find all single items whose support is greater than or equal to minimum support.

### Step 2: Generate candidate 2-itemsets

Join frequent 1-itemsets to form candidate pairs.

### Step 3: Prune infrequent candidates

Remove itemsets whose support is below the threshold.

### Step 4: Generate rules

For each frequent itemset, generate rules and keep only those whose confidence is above minimum confidence.

## Application In This Project

Each row of the phishing dataset is converted into transaction form:

```text
T = {
having_IP_Address=-1,
URL_Length=1,
SSLfinal_State=-1,
URL_of_Anchor=-1,
Result=-1
}
```

The algorithm then finds frequent patterns such as:

```text
URL_of_Anchor=-1 => Result=-1
SSLfinal_State=-1 => Result=-1
SSLfinal_State=1 => Result=1
```

## Why Apriori Was Chosen

Apriori was chosen because:

1. The dataset features are categorical and easy to convert into transactions.
2. The project requires frequent pattern generation.
3. Apriori is a standard and academically accepted algorithm for association rule mining.
4. It generates interpretable rules that help explain phishing behavior.

## Conclusion

Apriori is useful in this project because it complements the ANN classifier. ANN predicts whether a website is phishing, while Apriori explains which feature combinations are strongly associated with phishing or legitimate websites.
