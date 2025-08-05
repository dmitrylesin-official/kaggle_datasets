# 🏠 Housing Prices Analysis & Prediction

📥 **You can download the dataset from Kaggle:**  
[🔗 Housing Price Data](https://www.kaggle.com/competitions/home-data-for-ml-course)

This project focuses on analyzing and predicting housing prices using real estate data. We perform data preprocessing, feature engineering, and apply machine learning models — primarily Gradient Boosting — to estimate property prices.

---

## 📂 Dataset

The dataset is loaded from a CSV file named **Housing.csv**, which contains various features of residential properties such as:

• Number of bedrooms and bathrooms

• Presence of guestroom, basement, hot water heating, air conditioning

• Location features like proximity to the main road or preferred area

• Furnishing status (later dropped)

• Target variable: **price**

---

## 🧹 Data Preprocessing & Logging
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    filename='log.txt',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    df = pd.read_csv('Housing.csv')
    logging.info('Dataset loaded successfully')
except Exception as e:
    logging.error('Failed to load dataset: %s', str(e))

```
• Dropped furnishingstatus: This feature was removed because it did not significantly impact the model's predictive performance.

• Binary encoding: Converted categorical "yes/no" columns into boolean True/False for compatibility with ML algorithms.
```python
cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[cols] = df[cols].replace({'yes': True, 'no': False})
```

##🏐 Feature Engineering

Created a new feature to better represent total space:
```python
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
```
Also, outliers above the 99th percentile in price were removed:
```python
df = df[df['price'] <= df['price'].quantile(0.99)]
```

---

## 📊 Data Visualization

### Histogram with Outlier Thresholds
To detect and eliminate outliers, we visualized the distribution of housing prices with vertical lines marking the 95th and 99th percentiles:
```python
plt.subplots(figsize=(10, 7))
sns.histplot(df['price'], label='Price')
plt.axvline(df['price'].quantile(0.95), label='0.95%', c='green')
plt.axvline(df['price'].quantile(0.99), label='0.99%', c='red')
plt.legend()
plt.title('Price Distribution with Outlier Thresholds')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()
```
![image](https://github.com/user-attachments/assets/2ecb4377-a7d8-4d5e-9a99-2be763353168)

Values above the 99th percentile were considered outliers and removed to prevent skewing the model:
```python
df = df[df['price'] <= df['price'].quantile(0.99)]
```
### Correlation Matrix
A heatmap was used to show correlation between numerical features:
```python
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm');
```

---

## 🧐 Model Building & Evaluation

We use GradientBoostingRegressor and GridSearchCV for tuning:
```python
params = [{
    'n_estimators': [30, 50, 70],
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [2, 5]
}]
```
• 80/20 train/test split

• Scoring metric: R² score
```python
from sklearn.metrics import r2_score
y_pred = best_model.predict(X_test)
print(f'R² Score on test set: {r2_score(y_test, y_pred)}')
```
Logging is integrated throughout model training:
```python
logging.info("Starting GridSearchCV with parameters: %s", params)
...
logging.info("Test set R² score: %.4f", final_r2)
```

---

## 📈 Final Result

The model achieved an average R² score of ~0.67 on the test set, indicating a moderately strong relationship between the features and housing prices.

---

## 🛠 Technologies Used

• Python

• Pandas, NumPy for data manipulation

• Matplotlib, Seaborn for visualization

• Scikit-learn (sklearn) for model training and evaluation

• Logging — tracking data load, training, errors

• Google Colab as the development environment

---

## 📬 Author

Telegram: @dmitrylesin

Email: dmitrylesin_official@gmail.com

© 2025 Dmitry Lesin. All rights reserved.

