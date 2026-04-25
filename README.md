# Webpage Knowledge Discovery Using Clustering, Regression, Neural Networks, and Recommendation

This project uses the UCI phishing websites dataset to build a four-component machine learning review project:

1. Clustering to discover groups of webpages with similar behavior.
2. Regression to predict webpage traffic from the remaining website signals.
3. Neural networks to classify websites as phishing or legitimate.
4. A recommendation system to retrieve similar webpages for risk-aware analysis.

The repository is structured for project review, GitHub submission, and PPT preparation around one dataset and four required concepts.

## Dataset

- Source file: `dataset/uci-ml-phishing-dataset (1).csv`
- Records: 11,055
- Features: 30 predictive website features plus `Result`
- Target values:
  - `1` = legitimate website
  - `-1` = phishing website

## Project Structure

- `ml/data_utils.py` - shared dataset paths and loaders
- `ml/clustering_webpages.py` - K-Means clustering and cluster profiling
- `ml/regression_web_traffic.py` - Random forest regression for `web_traffic`
- `ml/deep_learning_phishing.py` - neural network classification
- `ml/recommendation_system.py` - content-based similar webpage recommendation
- `ml/random_forest_phishing.py` - optional baseline classifier
- `ml/association_rule_mining.py` - previous exploratory script, not part of the required four components
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

## Run The Four Components

```powershell
python ml\clustering_webpages.py
python ml\regression_web_traffic.py
python ml\deep_learning_phishing.py
python ml\recommendation_system.py --query-index 0 --top-k 5
```

Representative outcomes observed on this dataset are approximately:

- Clustering: best silhouette score near `0.2845` at `k = 3`
- Regression (`web_traffic`): MAE `0.3211`, RMSE `0.5102`, R2 `0.6136`
- Neural network: Accuracy `0.9692`, Precision `0.9671`, Recall `0.9781`, F1-score `0.9725`, ROC-AUC `0.9962`
- Recommendation: same-class neighbor ratio about `0.9131`

## Output Files

Running the scripts creates:

- `outputs/clustering_k_selection.csv`
- `outputs/clustering_summary.csv`
- `outputs/clustering_feature_means.csv`
- `outputs/regression_metrics.csv`
- `outputs/regression_cv_metrics.csv`
- `outputs/regression_feature_importance.csv`
- `outputs/regression_sample_predictions.csv`
- `outputs/neural_network_metrics.csv`
- `outputs/neural_network_cv_metrics.csv`
- `outputs/neural_network_confusion_matrix.csv`
- `outputs/recommendations.csv`
- `outputs/recommendation_metrics.csv`

## Suggested PPT Flow

Use the content in `docs/presentation_outline.md` for your review PPT:

1. Title
2. Problem statement
3. Dataset description
4. Clustering component
5. Regression component
6. Neural network component
7. Recommendation system component
8. Comparative findings
9. GitHub repository link
10. Conclusion and future work

## GitHub Upload Steps

Initialize git and push to GitHub after creating an empty repository in your account:

```powershell
git init
git add .
git commit -m "Add four-component webpage knowledge discovery project"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

Add the repository URL to your PPT on a dedicated slide or in the footer of the final slide.
