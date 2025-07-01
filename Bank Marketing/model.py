# Installing libraries (if not already installed)
# For local Python environment:
# pip install pandas numpy catboost scikit-learn

# For Google Colab, add "!" at the beginning of each command:
# !pip install pandas numpy catboost scikit-learn

import pandas as pd
import numpy as np
from collections import Counter

from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

import logging

logging.basicConfig(
    level=logging.INFO,
    filename='log.txt',
    filemode='a',
    format='%(astime)s - %(levelname)s - %(message)s'
)

logging.info('Dataset loaded successfully. Rows: %d, Columns: %d', df.shape[0], df.shape[1])
df = pd.read_csv('bank-additional-full.csv', sep=';', encofing='utf-8')

df.head()
df.info()
df.isnull().sum()

# Key: convert selected binary categorical columns to numeric
cols_to_encode = ['loan', 'housing']
for col in cols_to_encode:
    df[col] = df[col].map({'no': 0, 'yes': 1, 'unknown': 'unknown'})

# Key: convert target to boolean
df['y'] = df['y'].replace({'yes': True, 'no': False})

# Drop non-informative column
df = df.drop(columns='default')

X = df.drop(columns='y')
y = df['y']

logging.info('Split into train and test. Train size: %d, Test size: %d', len(X_train), len(X_test))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Key: identify categorical features for CatBoost
object_columns = df.select_dtypes(include='object').columns.tolist()
train_pool = Pool(data=X_train, label=y_train, cat_features=object_columns)
test_pool = Pool(data=X_test, label=y_test, cat_features=object_columns)

# Key: handle class imbalance using class weights
counts = Counter(y_train)
total = sum(counts.values())
class_weights = {
    0: total / (2 * counts[0]),
    1: total / (2 * counts[1]),
}

logging.info('Training model: CatBoostClassifier')
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=8,
    eval_metric='Recall',
    random_state=42,
    verbose=100,
    class_weights=class_weights
)

# Train the model with early stopping
model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

# Predictions
y_pred = model.predict(test_pool)
y_pred_proba = model.predict_proba(test_pool)[:, 1]

# Evaluation (Recall is the focus)
print("Recall:", recall_score(y_test, y_pred))
logging.info(f'Model evaluation complete. Recall: {recall:.4f}')
