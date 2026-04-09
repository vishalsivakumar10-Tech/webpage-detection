# Phishing Website Detection Using Deep Learning and Association Rule Mining

This project uses the UCI phishing websites dataset to build:

1. A deep learning based phishing detector using a multi-layer perceptron (ANN).
2. An Association Rule Mining workflow using the Apriori algorithm to discover frequent phishing patterns.

The repository is structured to support a semester project review, GitHub submission, and PPT preparation.

## Dataset

- Source file: `dataset/uci-ml-phishing-dataset (1).csv`
- Records: 11,055
- Features: 30 predictive website features plus `Result`
- Target values:
  - `1` = legitimate website
  - `-1` = phishing website

## Project Structure

- `ml/random_forest_phishing.py` - existing baseline model
- `ml/deep_learning_phishing.py` - ANN training, testing, and evaluation
- `ml/association_rule_mining.py` - frequent pattern generation and rule evaluation
- `docs/presentation_outline.md` - slide-by-slide PPT content
- `requirements.txt` - Python package requirements
- `.gitignore` - ignores generated artifacts

## Installation

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run The Deep Learning Model

```powershell
python ml\deep_learning_phishing.py
```

The script performs:

- data loading and preprocessing
- train/test split
- ANN training with early stopping
- test-set evaluation
- 5-fold cross-validation
- confusion matrix export
- metrics export to CSV

Expected performance from the configured model on this dataset is approximately:

- Test Accuracy: `0.9692`
- Precision: `0.9671`
- Recall: `0.9781`
- F1-score: `0.9725`
- ROC-AUC: `0.9962`

## Run Association Rule Mining

```powershell
python ml\association_rule_mining.py
```

The script performs:

- transaction generation from categorical phishing features
- frequent itemset mining with Apriori
- association rule generation
- rule quality measurement using support, confidence, lift, leverage, conviction
- test-set validation of discovered rules

Representative strong rules from this dataset include:

- `URL_of_Anchor=-1 => Result=-1`
- `SSLfinal_State=-1 => Result=-1`
- `SSLfinal_State=1 => Result=1`

## Output Files

Running the scripts creates:

- `outputs/deep_learning_metrics.csv`
- `outputs/deep_learning_cv_metrics.csv`
- `outputs/deep_learning_confusion_matrix.csv`
- `outputs/frequent_itemsets.csv`
- `outputs/association_rules.csv`

## Suggested PPT Flow

Use the content in `docs/presentation_outline.md` for your review PPT:

1. Title
2. Problem statement
3. Dataset description
4. Methodology
5. Deep learning model
6. Deep learning results
7. Association rule mining model
8. Frequent pattern results
9. GitHub repository link
10. Conclusion and future work

## GitHub Upload Steps

Initialize git and push to GitHub after creating an empty repository in your account:

```powershell
git init
git add .
git commit -m "Add deep learning and association rule mining project"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

Add the repository URL to your PPT on a dedicated slide or in the footer of the final slide.
