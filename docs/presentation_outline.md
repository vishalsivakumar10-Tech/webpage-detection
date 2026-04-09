# PPT Content For Semester Project Review

## Slide 1: Title

**Title:** Phishing Website Detection Using Deep Learning and Association Rule Mining

- Student name
- Register number
- Department / semester
- Guide name

## Slide 2: Problem Statement

- Phishing websites imitate trusted websites to steal user credentials and private data.
- Traditional detection methods are not always enough for identifying new phishing patterns.
- The objective is to classify websites as phishing or legitimate and discover hidden patterns from the same dataset.

## Slide 3: Objectives

- Build a deep learning model for phishing website detection.
- Train and test the model on the phishing dataset.
- Measure model performance using standard classification metrics.
- Generate frequent patterns and rules using Association Rule Mining.
- Evaluate the quality of generated rules.

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

## Slide 5: Methodology

- Data loading and preprocessing
- Train-test split
- Deep learning model training using ANN
- Model testing and metric calculation
- Frequent itemset generation using Apriori
- Association rule generation and rule evaluation

## Slide 6: Deep Learning Model

**Technique:** Multi-Layer Perceptron (Artificial Neural Network)

- Input layer: phishing website features
- Hidden layers: `64`, `32`, `16`
- Activation: `ReLU`
- Optimizer: `Adam`
- Early stopping used to prevent overfitting
- Target mapping:
  - phishing = `0`
  - legitimate = `1`

## Slide 7: Deep Learning Performance

Use these results from the implemented model:

- Accuracy: `96.92%`
- Precision: `96.71%`
- Recall: `97.81%`
- F1-score: `97.25%`
- ROC-AUC: `99.62%`

5-fold cross-validation average:

- Accuracy: `96.25%`
- Precision: `96.37%`
- Recall: `96.91%`
- F1-score: `96.64%`
- ROC-AUC: `99.41%`

Confusion matrix:

- True phishing detected: `939`
- Legitimate predicted as phishing: `41`
- Phishing predicted as legitimate: `27`
- True legitimate detected: `1204`

## Slide 8: Association Rule Mining

**Technique:** Apriori Algorithm

- Transactions were formed by converting each feature-value pair into an item.
- Minimum support: `20%`
- Minimum confidence: `80%`
- Measures used:
  - support
  - confidence
  - lift
  - leverage
  - conviction

## Slide 9: Important Frequent Patterns And Rules

Strong rules obtained:

1. `URL_of_Anchor=-1 => Result=-1`
   - Support: `29.51%`
   - Confidence: `98.90%`
   - Lift: `2.23`

2. `SSLfinal_State=-1 => Result=-1`
   - Support: `27.75%`
   - Confidence: `85.62%`
   - Lift: `1.93`

3. `SSLfinal_State=1 => Result=1`
   - Support: `50.87%`
   - Confidence: `89.02%`
   - Lift: `1.60`

4. `URL_of_Anchor=1 => Result=1`
   - Support: `20.60%`
   - Confidence: `93.87%`
   - Lift: `1.69`

## Slide 10: Association Rule Performance

- Number of strong rules generated: `5`
- Highest confidence rule:
  - `URL_of_Anchor=-1 => Result=-1`
  - Test confidence: `98.91%`
- Highest coverage rule:
  - `SSLfinal_State=1 => Result=1`
  - Test coverage: `57.76%`

## Slide 11: Tools And Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Apriori-based pattern mining
- GitHub for code hosting
- Microsoft PowerPoint / Google Slides

## Slide 12: GitHub Repository

Add your repository URL here:

`https://github.com/<your-username>/<your-repository-name>`

Suggested note:

- Full source code, dataset reference, outputs, and documentation are available in the GitHub repository.

## Slide 13: Conclusion

- The ANN model achieved high accuracy and ROC-AUC for phishing website detection.
- Association rule mining exposed interpretable phishing indicators from website features.
- Combining prediction and pattern discovery makes the project stronger for academic review.

## Slide 14: Future Scope

- Try TensorFlow or PyTorch models for deeper architectures.
- Deploy the classifier as a web application.
- Use live URL feature extraction.
- Add SHAP or feature interpretation for explainability.

## Slide 15: Demo Plan

During the review, show:

1. Dataset file
2. Deep learning code
3. Association rule mining code
4. Generated output CSV files
5. GitHub repository link
6. Final result slides
