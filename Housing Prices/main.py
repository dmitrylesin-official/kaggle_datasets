# 📦 Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 📥 Load the dataset
df = pd.read_csv('Housing.csv')

# 🔍 Display first few rows and dataset info
df.head()
df.info()

# 🧼 Check for missing values
df.isnull().sum()

# 🧹 Drop the 'furnishingstatus' column — it does not contribute to prediction
df = df.drop('furnishingstatus', axis=1)

# 🔄 Convert categorical 'yes/no' columns to boolean values
cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[cols] = df[cols].replace({'yes': True, 'no': False})

# 🧮 Create a new feature combining bedrooms and bathrooms
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

# 🧪 Split the dataset into features and target variable
X = df.drop('price', axis=1)
y = df['price']

# 📊 Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔧 Define hyperparameters for grid search
params = [{
    'n_estimators': [30, 50, 70],
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [2, 5]
}]

# 🧠 Initialize the Gradient Boosting model
model = GradientBoostingRegressor(random_state=42)

# 🔍 GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=5,
    scoring='r2'
)

# 🚂 Train the model on the training data
grid_search.fit(X_train, y_train)

# 🏆 Show best hyperparameters and cross-validation score
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best cross-validation score: {grid_search.best_score_}')

# ✅ Select the best model from grid search
best_model = grid_search.best_estimator_

# 📈 Predict on test set and evaluate performance
y_pred = best_model.predict(X_test)
print(f'R² Score on test set: {r2_score(y_test, y_pred)}')

# 🔥 Visualize feature correlation matrix
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation with Price')
plt.show()
