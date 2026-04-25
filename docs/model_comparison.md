# Model Comparison

## Objective

This project focuses on phishing website detection using a deep learning model and interpretable pattern mining. A model comparison is useful to justify why the Artificial Neural Network (ANN) was selected as the main predictive model.

## Models Considered

### 1. Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees and gives the final prediction based on majority voting.

**Advantages**

- Works well on structured tabular data
- Handles non-linear decision boundaries
- Provides feature importance
- Less sensitive to overfitting than a single decision tree

**Limitations**

- Decision boundaries are rule-based, not representation-learning based
- Harder to extend into deep learning justification for an academic review
- Less suitable when the project explicitly asks for a deep learning technique

### 2. Artificial Neural Network (ANN)

ANN is a deep learning model made of interconnected layers of neurons. It learns weights and biases during training to capture relationships between input features and output classes.

**Advantages**

- Learns complex non-linear relationships
- Captures interactions among many website features
- Matches the requirement of using a deep learning technique
- Produces strong classification performance on this dataset

**Limitations**

- Less interpretable than tree-based models
- Needs feature scaling and tuning
- Training can be slower than traditional machine learning models

## Why ANN Was Chosen

The semester task explicitly required a deep learning technique. ANN was selected because:

1. The phishing dataset contains many website indicators that interact in a non-linear way.
2. ANN can learn hidden feature combinations through multiple layers.
3. ANN achieved strong performance on the dataset.
4. ANN demonstrates a modern AI-based approach, which strengthens the academic value of the project.

## Comparison Summary

| Model | Type | Strength | Weakness | Project Role |
|---|---|---|---|---|
| Random Forest | Traditional ML | Strong baseline, interpretable feature importance | Not a deep learning model | Baseline comparison |
| ANN | Deep Learning | Learns complex patterns, high predictive performance | Lower interpretability | Main classification model |

## Performance Summary

The ANN model achieved the following approximate test performance:

- Accuracy: 96.92%
- Precision: 96.71%
- Recall: 97.81%
- F1-score: 97.25%
- ROC-AUC: 99.62%

These results justify the selection of ANN as the final predictive model.

## Final Justification

Random Forest is a strong baseline for phishing detection, but ANN was chosen as the final model because the project specifically required deep learning and because ANN can model complex feature interactions more effectively in a layered architecture.
