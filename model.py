import pandas as pd
import logging

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Pipeline started.")

# Load dataset
df = pd.read_csv('Employee.csv')
logging.info("Dataset loaded successfully.")

# Drop duplicates
duplicates = df.duplicated().sum()
logging.info(f"Duplicate rows found: {duplicates}. Removing them.")
df = df.drop_duplicates()

# Rename columns for clarity
df.columns = [
    'education', 'joining_year', 'city', 'payment_tier', 'age',
    'gender', 'ever_benched', 'experience_in_current_domain', 'leave_or_not'
]

# Convert year to string (categorical feature)
df['joining_year'] = df['joining_year'].astype(str)

# Split features and target
X = df.drop('leave_or_not', axis=1)
y = df['leave_or_not']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
logging.info("Train-test split completed.")

# Define categorical and numerical features
cat_features = ['education', 'joining_year', 'city', 'gender', 'ever_benched']
num_features = ['age', 'payment_tier', 'experience_in_current_domain']

# Pipeline for categorical features
cat_pipeline = Pipeline([
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

# Pipeline for numerical features
num_pipeline = Pipeline([
    ('scale', StandardScaler())
])

# Combine pipelines
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Full model pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

# Define hyperparameter grid
params = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [10, 20, 30],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 3]
}

# Perform GridSearchCV
logging.info("Starting GridSearchCV...")
grid_search = GridSearchCV(
    model_pipeline,
    param_grid=params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Train the model
grid_search.fit(X_train, y_train)
logging.info("Model training completed.")

# Predict on test data
y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Test Accuracy: {accuracy:.2f}")

# Output best parameters
logging.info("Best parameters:")
logging.info(grid_search.best_params_)
