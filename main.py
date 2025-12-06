import os
import pandas as pd

import kagglehub
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Path to dataset files:", path)
print("Files in this folder:", os.listdir(path))

data_path = Path(path) / "creditcard.csv"
df = pd.read_csv(data_path)

print(f"Dataframe's shape: {df.shape}")
print(f"Dataframe's dtypes: {df.dtypes}")
print(f"Dataframe's head: {df.head()}")
print(f"Dataframe's class counts: {df['Class'].value_counts(normalize=True)}")
print(f"Dataframe's NaN: {df.isna().mean().sort_values(ascending=False)}")

X = df.drop('Class', axis=1)
y = df['Class']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(x_train, y_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

x_train_small, _, y_train_small, _ = train_test_split(
    x_train_scaled,
    y_train,
    train_size=25000,
    random_state=42,
    stratify=y_train
)

log_res = LogisticRegression(max_iter=1000)
param_grid_log_reg = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'class_weight': [None, 'balanced']
}
grid_log = GridSearchCV(
    estimator=log_res,
    param_grid=param_grid_log_reg,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=1
)
grid_log.fit(x_train_small, y_train_small)
y_pred_log = grid_log.predict(x_test_scaled)
print(f"LogisticRegression Classification Report: {classification_report(y_test, y_pred_log)}")
print(f"LogisticRegression Confusion Matrix: {confusion_matrix(y_test, y_pred_log)}\n")

dt = DecisionTreeClassifier(random_state=42)
param_grid_dt = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 10, 50],
    'min_samples_leaf': [1, 5, 10],
    'max_features': [None, 'sqrt', 'log2'],
    'class_weight': [None, 'balanced']
}
grid_dt = GridSearchCV(
    estimator=dt,
    param_grid=param_grid_dt,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=1
)
grid_dt.fit(x_train_small, y_train_small)
y_pred_dt = grid_dt.predict(x_test_scaled)
print(f"DecisionTree Classification Report: {classification_report(y_test, y_pred_dt)}")
print(f"DecisionTree Confusion Matrix: {confusion_matrix(y_test, y_pred_dt)}\n")

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)
param_grid_rf = {
    'n_estimators': [100, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 5],
    'max_features': ['sqrt', 'log2'],
    'class_weight': [None, 'balanced']
}
grid_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_rf,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=1
)
grid_rf.fit(x_train_small, y_train_small)
y_pred_rf = grid_rf.predict(x_test_scaled)
print(f"RandomForest Classification Report: {classification_report(y_test, y_pred_rf)}")
print(f"RandomForest Confusion Matrix: {confusion_matrix(y_test, y_pred_rf)}\n")

y_proba_log = grid_log.predict_proba(x_test_scaled)[:, 1]
y_proba_dt = grid_dt.best_estimator_.predict_proba(x_test_scaled)[:, 1]
y_proba_rf = grid_rf.best_estimator_.predict_proba(x_test_scaled)[:, 1]

print("LogReg ROC-AUC:", roc_auc_score(y_test, y_proba_log))
print("DecisionTree ROC-AUC:", roc_auc_score(y_test, y_proba_dt))
print("RandomForest ROC-AUC:", roc_auc_score(y_test, y_proba_rf))
