# PPT Content For Semester Project Review

## Slide 1: Title

**Title:** Webpage Knowledge Discovery Using Clustering, Regression, Neural Networks, and Recommendation

- Student name
- Register number
- Department / semester
- Guide name

## Slide 2: Problem Statement

- Webpages have multiple observable signals such as URL structure, SSL state, traffic, and domain behavior.
- These signals can be used not only for classification but also for segmentation, prediction, and similarity-based retrieval.
- The objective is to use one dataset to demonstrate four machine learning concepts required for project review.

## Slide 3: Objectives

- Apply clustering to group webpages with similar feature behavior.
- Build a regression model to predict `web_traffic`.
- Build a neural network to classify phishing and legitimate websites.
- Build a recommendation system to retrieve similar webpages.
- Compare what each component contributes to knowledge discovery.

## Slide 4: Dataset Description

- Dataset: UCI Phishing Websites Dataset
- Total records: 11,055
- Features: 30 website-based features
- Class label:
  - `1` = legitimate
  - `-1` = phishing
- Feature examples:
  - `having_IP_Address`
  - `URL_Length`
  - `SSLfinal_State`
  - `URL_of_Anchor`
  - `web_traffic`

## Slide 5: Clustering Methodology

- Data preprocessing using imputation and scaling
- K-Means clustering tested for multiple `k` values
- Silhouette score used for cluster selection
- Cluster profiles interpreted using phishing ratio, page rank, and web traffic

## Slide 6: Clustering Results

- Best silhouette score observed near `0.2845`
- Best number of clusters: `3`
- Each cluster represents a different webpage behavior segment
- Use cluster summaries from `clustering_summary.csv`

## Slide 7: Regression Methodology

- Target variable: `web_traffic`
- Model: Random Forest Regressor
- Input: all remaining webpage features
- Evaluation metrics: MAE, RMSE, and R2
- Cross-validation used for stability checking

## Slide 8: Regression Results

- MAE: `0.3211`
- RMSE: `0.5102`
- R2: `0.6136`
- Important predictors generally include security and ranking related features
- Use `regression_feature_importance.csv` for charts

## Slide 9: Neural Network Methodology

- Technique: Multi-Layer Perceptron
- Hidden layers: `64`, `32`, `16`
- Activation: `ReLU`
- Optimizer: `Adam`
- Early stopping used to reduce overfitting
- Output: phishing vs legitimate webpage classification

## Slide 10: Neural Network Results

- Accuracy: `96.92%`
- Precision: `96.71%`
- Recall: `97.81%`
- F1-score: `97.25%`
- ROC-AUC: `99.62%`
- Show confusion matrix from `neural_network_confusion_matrix.csv`

## Slide 11: Recommendation System Methodology

- Approach: content-based recommendation using nearest neighbors
- Similarity metric: cosine similarity
- Input: preprocessed webpage feature vectors
- Output: most similar webpages for a given query page
- Evaluation: same-class neighbor ratio

## Slide 12: Recommendation Results

- Same-class neighbor ratio: `91.31%`
- Similar webpages usually share the same phishing or legitimate label
- Recommendation can support analysts by surfacing look-alike cases
- Show sample recommendations from `recommendations.csv`

## Slide 13: Comparative Insight

- Clustering explains hidden segments in the data
- Regression predicts an operational numeric signal
- Neural networks deliver the strongest predictive performance
- Recommendation adds similarity-based decision support

## Slide 14: Tools And Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- GitHub for code hosting
- Microsoft PowerPoint / Google Slides

## Slide 15: GitHub Repository

Add your repository URL here:

`https://github.com/<your-username>/<your-repository-name>`

Suggested note:

- Full source code, dataset reference, outputs, and documentation are available in the GitHub repository.

## Slide 16: Conclusion

- The same webpage dataset supports four different machine learning tasks.
- The neural network gives strong classification performance.
- Regression and clustering provide additional analytical understanding.
- Recommendation adds practical similarity-based support for analysts.

## Slide 17: Future Scope

- Try DBSCAN or hierarchical clustering for different segments.
- Use gradient boosting for regression comparison.
- Deploy the classifier and recommender in a web interface.
- Add explainability such as SHAP for model interpretation.

## Slide 18: Demo Plan

During the review, show:

1. Dataset file
2. Clustering code
3. Regression code
4. Neural network code
5. Recommendation code
6. Generated output CSV files
7. GitHub repository link
8. Final result slides
