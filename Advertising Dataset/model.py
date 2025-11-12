# Installing required libraries (if not installed yet)
# For local environment (desktop Python):
# pip install pandas scikit-learn

# For Google Colab, add "!" before each command:
# !pip install pandas scikit-learn

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

import logging

import joblib

logging.basicConfig(
    level=logging.INFO,
    filename='log.txt',
    filemode='a',
    format='%(astime)s - %(levelname)s - %(message)s'
)

logging.info('Dataset loaded successfully.')
df = pd.read_csv('advertising.csv', encoding='utf-8')

df.head()
df.info()
df.isnull().sum()
df.duplicated().sum()

# Separate features and target variable
X = df.drop('Sales', axis=1)
y = df['Sales']

logging.info('Split into train and test. Train size: %d, Test size: %d', len(X_train), len(X_test))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a pipeline:
# Step 1: Add polynomial features (degree=2)
# Step 2: Apply Gradient Boosting Regressor
pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', GradientBoostingRegressor(random_state=42))
])

# Set up hyperparameter grid for tuning the model
param_grid = {
    'model__n_estimators': [30, 50, 70],
    'model__max_depth': [3, 5, 7],
    'model__min_samples_split': [2, 4]
}

# Use GridSearchCV to find the best combination of hyperparameters
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='r2'  # R² score as the evaluation metric
)

# Fit the pipeline to the training data
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Print best hyperparameters and evaluation metrics
print('Best params:', grid_search.best_params_)
print('Best R2 (CV):', grid_search.best_score_)
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}')

joblib.dump(best_model, 'model.pkl')
