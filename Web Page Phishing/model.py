import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

from xgboost import XGBClassifier, plot_importance

from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

import logging

# Configure logging to file
logging.basicConfig(
    level=logging.INFO,
    filename='log.txt',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Log dataset load
logging.info('Dataset loaded successfully')

# Load dataset
df = pd.read_csv('web-page-phishing.csv')

# Inspect structure and missing values
df.info()
df.head()
df.isnull().sum()

# Plot class distribution
df['phishing'].hist()

# Plot correlation heatmap
plt.subplots(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Define features and target
X = df.drop('phishing', axis=1)
y = df['phishing']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Handle class imbalance
counts = Counter(y_train)
scale_pos_weight = counts[0] / counts[1]

# Initialize and train XGBoost model
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight
)
model.fit(X_train, y_train)

results = model.get_evals_result()

plt.figure(figsize=(10, 6))
plt.plot(results['learn']['Logloss'], label='Train')
plt.plot(results['validation']['Logloss'], label='Validation')
plt.xlabel('Iteration')
plt.ylabel('Logloss')
plt.title('CatBoost Learning Curve')
plt.legend()
plt.grid(True)
plt.show()

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model using Recall
print(f'Recall: {recall_score(y_test, y_pred)}')

# Plot top important features
plot_importance(model, max_num_features=20, importance_type='gain')
plt.show()
