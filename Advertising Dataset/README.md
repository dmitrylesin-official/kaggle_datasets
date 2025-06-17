# 📈 Advertising Sales Prediction

📥 You can download the dataset from Kaggle:  
🔗 [Advertising Dataset](https://www.kaggle.com/datasets/ashydv/advertising-dataset)

This project aims to analyze and predict advertising-driven product sales using classical machine learning techniques. We utilize polynomial feature expansion and Gradient Boosting to improve prediction accuracy.

---

## 📂 Dataset  
The dataset is loaded from a CSV file named `advertising.csv`, which includes:

• TV advertising budget  
• Radio advertising budget  
• Newspaper advertising budget  
• Target variable: **Sales**

---

## 🧹 Data Preprocessing

```python
df = pd.read_csv('/content/advertising.csv')
```

• We read the dataset using pandas and explored the basic statistics (.info() and .head()).

• The target variable Sales was separated from the features.


```python
X = df.drop('Sales', axis=1)
y = df['Sales']
```

---

## 📊 Data Exploration: Histograms
To better understand the distribution of the advertising budgets and the sales values, histograms were plotted for each feature:

• Histograms help visualize the frequency distribution of each variable.

• They reveal important characteristics such as skewness, range, and the presence of outliers.

• This initial exploration guides us in preprocessing decisions and feature engineering.
```python
df.hist();
```
![image](https://github.com/user-attachments/assets/f94a64d7-858b-4e1d-aba7-b5d1602d525e)

From the histograms, we can observe:

• The TV and Radio budgets show roughly normal distributions with some variation.

• Newspaper budgets tend to be skewed with many low values and a few high spenders.

• Sales data is approximately normally distributed, indicating consistent sales behavior with some variation.

These insights help justify applying polynomial features to model nonlinear relationships between advertising spend and sales.

---

## Machine Learning Pipeline

We use a pipeline to chain together:

1. PolynomialFeatures (degree=2) — expands features with squared and interaction terms

2. GradientBoostingRegressor — for prediction

```python
pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', GradientBoostingRegressor(random_state=42))
])
```

---

## 🎯 Hyperparameter Tuning with GridSearchCV

To find the best model configuration, we apply GridSearchCV with 5-fold cross-validation:

```python
param_grid = {
    'model__n_estimators': [30, 50, 70],
    'model__max_depth': [3, 5, 7],
    'model__min_samples_split': [2, 4]
}

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring='r2'
)
```

• Data split: 80% for training, 20% for testing

• Evaluation metrics: Mean Squared Error (MSE) and Mean Absolute Error (MAE)

---

## 📈 Final Result

After fitting the pipeline:
```python
grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)
```
We evaluate the model using standard regression metrics:
```python
print('Best params:', grid_search.best_params_)
print('Best R2 (CV):', grid_search.best_score_)
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
```
📌 Output Example (may vary based on run):
```yaml
Best params: {'model__max_depth': 3, 'model__min_samples_split': 2, 'model__n_estimators': 70}
Best R2 (CV): 0.974
MSE: 1.05
MAE: 0.67
```

---

## 🛠 Technologies Used

**• Python**

**• Pandas** — for data manipulation

**• Scikit-learn (sklearn)** — for modeling and evaluation

**• PolynomialFeatures** — to create interaction features

**• GridSearchCV** — for hyperparameter tuning

**• Google Colab** — as the development environment

---

## 📬 Author

**Telegram:** @dmitrylesin

**Email:** dmitrylesin_official@gmail.com

**© 2025 Dmitry Lesin. All rights reserved.**
