# Installing required libraries (if not installed yet)
# For local Python environment:
# pip install pandas scikit-learn matplotlib seaborn

# For Google Colab, add "!" at the beginning of each command:
# !pip install pandas scikit-learn matplotlib seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import logging

logging.basicConfig(
    level=logging.INFO,
    filename='log.txt',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    df = pd.read_csv('Housing.csv', encoding='utf-8')
    logging.info('Dataset loaded successfully')
except Exception as e:
    logging.error('Failed to load dataset: %s', str(e)

# Display first few rows and dataset info
df.head()
df.info()

# Check for missing values
df.isnull().sum()

logging.info("Dropped column 'furnishingstatus' as irrelevant.")
df = df.drop('furnishingstatus', axis=1)

# Convert categorical 'yes/no' columns to boolean values
cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[cols] = df[cols].replace({'yes': True, 'no': False})

# Visualize price distribution and cut off outliers
plt.subplots(figsize=(10, 7))
sns.histplot(df['price'], label='Price')
plt.axvline(df['price'].quantile(0.95), label='0.95%', c='green')
plt.axvline(df['price'].quantile(0.99), label='0.99%', c='red')
plt.legend()
plt.title('Price Distribution with Outlier Thresholds')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

logging.info("Created new feature 'total_rooms' = bedrooms + bathrooms")
df = df[df['price'] <= df['price'].quantile(0.99)]

# Create a new feature combining bedrooms and bathrooms
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

# Split the dataset into features and target variable
X = df.drop('price', axis=1)
y = df['price']

logging.info("Data split into train/test. Train: %d, Test: %d", len(X_train), len(X_test))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters for grid search
params = [{
    'n_estimators': [30, 50, 70],
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [2, 5]
}]

# Initialize the Gradient Boosting model
model = GradientBoostingRegressor(random_state=42)

logging.info("Starting GridSearchCV with parameters: %s", params)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=5,
    scoring='r2'
)

# Train the model on the training data
grid_search.fit(X_train, y_train)

logging.info("Best parameters found: %s", grid_search.best_params_)
logging.info("Best CV R² score: %.4f", grid_search.best_score_)
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best cross-validation score: {grid_search.best_score_}')

# Select the best model from grid search
best_model = grid_search.best_estimator_

final_r2 = r2_score(y_test, y_pred)
logging.info("Test set R² score: %.4f", final_r2)
y_pred = best_model.predict(X_test)
print(f'R² Score on test set: {r2_score(y_test, y_pred)}')

# Visualize feature correlation matrix
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation with Price')
plt.show()
