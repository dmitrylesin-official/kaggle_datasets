# 🧑‍💼 Employee Attrition Prediction
## 📥 Dataset
This project uses an internal HR dataset from a CSV file: **Employee.csv**, which includes various employee attributes like:

• Education level, joining year, city, gender
• Salary tier and age
• Whether they’ve ever been benched (temporarily removed from projects)
• Experience in the current domain
• Binary target variable: leave_or_not (1 = employee left, 0 = stayed)

---

## 🧹 Data Preprocessing
```python
df = pd.read_csv('Employee.csv')
df = df.drop_duplicates()
df.columns = ['education', 'joining_year', 'city', 'payment_tier', 'age',
              'gender', 'ever_benched', 'experience_in_current_domain', 'leave_or_not']
df['joining_year'] = df['joining_year'].astype(str)
```
• Duplicates were removed to avoid data leakage.
• Columns were renamed for clarity and uniformity.
• joining_year was converted to string to treat it as a categorical variable.

---

## ⚙️ Feature Engineering & Pipelines
We separated categorical and numerical features:
```python
cat_features = ['education', 'joining_year', 'city', 'gender', 'ever_benched']
num_features = ['age', 'payment_tier', 'experience_in_current_domain']
```
Created separate pipelines for numeric and categorical processing:
```python
cat_pipeline = Pipeline([
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

num_pipeline = Pipeline([
    ('scale', StandardScaler())
])
```
Then we combined them:
```python
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])
```
---

## 🧠 Model Training with GridSearchCV
We used **RandomForestClassifier** and performed hyperparameter tuning using **GridSearchCV**:
```python
params = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [10, 20, 30],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 3]
}
```
The pipeline includes preprocessing and model fitting:
```python
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

grid_search = GridSearchCV(model_pipeline, param_grid=params, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```
---

## 📈 Final Results
```yaml
Test Accuracy: 0.86
Best Parameters:
{'model__max_depth': 20, 'model__min_samples_leaf': 1, 'model__min_samples_split': 2, 'model__n_estimators': 100}
```

---

## 🛠 Technologies Used

• Python

• Pandas for data processing

• Scikit-learn for modeling and pipeline management

• GridSearchCV for hyperparameter tuning

• Logging for visibility and reproducibility

---

## 🧾 Logging System
We used Python’s built-in logging module to track key stages of the pipeline:
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```
Logged events include:

• Dataset loading

• Duplicate removal

• Model training

• Accuracy on test set

• Best hyperparameters found

This makes debugging and workflow tracing easier — especially for larger projects or in production environments.

---

## 📬 Author
Telegram: @dmitrylesin

Email: dmitrylesin_official@gmail.com

© 2025 Dmitry Lesin. All rights reserved.
