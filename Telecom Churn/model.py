# Installing required libraries (if not installed yet)
# For local Python environment:
# pip install pandas scikit-learn matplotlib seaborn

# For Google Colab, add "!" before each command:
# !pip install pandas scikit-learn matplotlib seaborn

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import logging

import joblib

logging.basicConfig(
    level=logging.INFO,
    filename='log.txt',
    filemode='a',
    format='%(asctime)s - %(levelname) - %(message)s'
)

logging.info('Dataset loaded successfully. Rows: %d, Columns: %d', df.shape[0], df.shape[1])
df = pd.read_csv('churn-bigml-20.csv', encoding='utf-8')

df.head()
df.info()
df.isnull().sum()
df.duplicated().sum()

# Drop the 'State' column (not useful for modeling)
df = df.drop('State', axis=1)

# Visualize distributions for all numeric features
df.hist(figsize=(14, 10))
plt.tight_layout()
plt.show()

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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

# Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Optional feature selection (not used in final model)
# selector = SelectKBest(score_func=f_classif, k=10)
# X_new = selector.fit_transform(X, y)

logging.info('Split into train and test. Train size: %d, Test size: %d', len(X_train), len(X_test))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = RandomForestClassifier(max_depth=15, n_estimators=100, random_state=42)
model_pipeline = Pipeline([
    ('model', RandomForestClassifier(max_depth=15, n_estimators=100, random_state=42))
])
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# Evaluate model performance
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(f'F1 Score: {f1_score(y_test, y_pred):.2f}')
logging.info(f'Model evaluation complete. Accuracy: {accuracy:.2f}, F1: {f1:.2f}')

dummy = DummyClassifier(strategy='stratified')
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
print(f'Recall: {recall_score(y_test, dummy_pred):.2f}')

joblib.dump(model_pipeline, 'model.pkl')
