# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('/content/churn-bigml-20.csv')

# Preview the first few rows
df.head()

# View data structure and types
df.info()

# Check for missing values
df.isnull().sum()

# Drop the 'State' column (not useful for modeling)
df = df.drop('State', axis=1)

# Visualize distributions for all numeric features
df.hist(figsize=(14, 10))
plt.tight_layout()
plt.show()

# Convert 'Yes'/'No' columns to boolean (True/False)
df['International plan'] = df['International plan'].map({'Yes': True, 'No': False})
df['Voice mail plan'] = df['Voice mail plan'].map({'Yes': True, 'No': False})

# Check distribution of 'Voice mail plan'
df['Voice mail plan'].value_counts()

# Correlation heatmap
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Drop low-impact or highly correlated features (based on correlation matrix and domain knowledge)
df = df.drop([
    'Total intl minutes',
    'Total night calls',
    'Total eve charge',
    'Total day charge',
    'Number vmail messages',
    'Total night charge'
], axis=1)

# Import machine learning tools
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Optional feature selection (not used in final model)
# selector = SelectKBest(score_func=f_classif, k=10)
# X_new = selector.fit_transform(X, y)

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(max_depth=15, n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'F1 Score: {f1_score(y_test, y_pred):.4f}')

# Optional: Hyperparameter tuning with GridSearchCV (worse results in this case)
# param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [10, 15, 20]
# }
# grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
# grid_search.fit(X_train, y_train)
# print("Best parameters:", grid_search.best_params_)
# y_pred_grid = grid_search.predict(X_test)
# print(f'Tuned Accuracy: {accuracy_score(y_test, y_pred_grid):.4f}')
# print(f'Tuned F1 Score: {f1_score(y_test, y_pred_grid):.4f}')

