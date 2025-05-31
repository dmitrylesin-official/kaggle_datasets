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

```df = pd.read_csv('/content/advertising.csv')```

• We read the dataset using pandas and explored the basic statistics (.info() and .head()).

• The target variable Sales was separated from the features.

```
X = df.drop('Sales', axis=1)
y = df['Sales']
```

---

## 🏐 Feature Engineering

We applied Polynomial Feature Expansion (degree=2) to model complex nonlinear interactions between the features:

```
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

---

## 🧐 Model Building & Evaluation

We used GradientBoostingRegressor from sklearn.ensemble, a powerful ensemble method for regression problems.

Hyperparameters were tuned using GridSearchCV for optimal performance:

```
params = [{
    'n_estimators': [30, 50, 70],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4]
}]
```

• Data split: 80% for training, 20% for testing

• Evaluation metrics: Mean Squared Error (MSE) and Mean Absolute Error (MAE)

```
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
