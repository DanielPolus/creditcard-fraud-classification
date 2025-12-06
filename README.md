# Credit Card Fraud Classification

Binary classification of credit card transactions using classical machine learning models to detect fraudulent operations.

## Dataset

- **Source:** Kaggle – [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions, 31 features
- **Target:** `Class` (0 – legitimate, 1 – fraud)
- **Imbalance:** fraud accounts for ~0.17% of all transactions

Data is downloaded programmatically using `kagglehub`.

## Models and Training

The project focuses on tabular ML with strong class imbalance:

- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**

For each model:

- Train/test split with stratification
- Standardization for linear model
- Hyperparameter tuning with `GridSearchCV` (3-fold CV, ROC-AUC as scoring)
- Evaluation on the held-out test set

## Results (fraud class only)

| Model              | Precision | Recall | F1-score |
|--------------------|-----------|--------|----------|
| Logistic Regression| 0.12      | 0.88   | 0.20     |
| Decision Tree      | 0.61      | 0.77   | 0.68     |
| Random Forest      | 0.88      | 0.65   | 0.75     |

- Logistic Regression acts as a very aggressive detector: high recall but many false positives.
- Decision Tree provides a more balanced trade-off between precision and recall.
- Random Forest achieves the best overall F1-score and high precision, making it the most conservative but reliable model.

## How to Run

```bash
git clone https://github.com/DanielPolus/creditcard-fraud-classification.git
cd creditcard-fraud-classification

# (optional) create and activate virtual environment

pip install -r requirements.txt
python main.py
