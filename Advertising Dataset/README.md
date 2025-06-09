# 📈 Advertising Sales Prediction

📥 You can download the dataset from Kaggle:  
🔗 [Advertising Dataset](https://www.kagglehttps://www.kaggle.com/datasets/ashydv/advertising-dataset)

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
From the histograms, we can observe:

• The TV and Radio budgets show roughly normal distributions with some variation.

• Newspaper budgets tend to be skewed with many low values and a few high spenders.

• Sales data is approximately normally distributed, indicating consistent sales behavior with some variation.

These insights help justify applying polynomial features to model nonlinear relationships between advertising spend and sales.

---

## 🏐 Feature Engineering

We applied Polynomial Feature Expansion (degree=2) to model complex nonlinear interactions between the features:

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

---

## 🧐 Model Building & Evaluation

We used GradientBoostingRegressor from sklearn.ensemble, a powerful ensemble method for regression problems.

Hyperparameters were tuned using GridSearchCV for optimal performance:

```python
params = [{
    'n_estimators': [30, 50, 70],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4]
}]
```

• Data split: 80% for training, 20% for testing

• Evaluation metrics: Mean Squared Error (MSE) and Mean Absolute Error (MAE)

```python
print(f'MSE: {mean_squared_error(y_test, y_pred)}') ≈ 0.980
print(f'MAE: {mean_absolute_error(y_test, y_pred)}') ≈ 0.672
```

---

## 📈 Final Result

The model was able to capture nonlinear trends in advertising effectiveness and produced a solid prediction quality based on MSE and MAE metrics. Grid search also showed optimal hyperparameters for boosting accuracy.

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

**Telegram:** @lesin_official

**Email:** dmitrylesin_official@gmail.com

**© 2025 Dmitry Lesin. All rights reserved.**
