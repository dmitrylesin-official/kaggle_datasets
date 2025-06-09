import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
df = pd.read_csv('advertising.csv')

# Show the first few rows of the dataframe
df.head()

# Display info about the dataframe (columns, data types, non-null counts)
df.info()

# Plot histograms for all numerical columns to understand distributions
df.hist();

# Separate features (X) and target variable (y)
X = df.drop('Sales', axis=1)
y = df['Sales']

# Generate polynomial features of degree 2 without bias term to capture nonlinear relationships
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Split the data into training and test sets (80% train, 20% test), with a fixed random seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting Regressor model
model = GradientBoostingRegressor(random_state=42)

# Define hyperparameter grid for tuning
params = [{
    'n_estimators': [30, 50, 70],        # Number of boosting stages
    'max_depth': [3, 5, 7],              # Maximum depth of each tree
    'min_samples_split': [2, 4]          # Minimum samples required to split an internal node
}]

# Setup GridSearchCV to find the best hyperparameters with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=5,
    scoring='r2'  # Use R-squared as the evaluation metric
)

# Fit the grid search on the training data to find the best parameters
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
print(grid_search.best_params_)

# Print the absolute value of the best R2 score from cross-validation
print(abs(grid_search.best_score_))

# Predict target values for the test set using the best model found
y_pred = grid_search.predict(X_test)

# Calculate and print Mean Squared Error on the test set
print(f'MSE: {mean_squared_error(y_test, y_pred)}')

# Calculate and print Mean Absolute Error on the test set
print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
