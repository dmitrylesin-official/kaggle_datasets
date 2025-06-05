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

## 🧹 Data Preprocessing
```python
df = pd.read_csv('/content/Housing.csv')
df = df.drop('furnishingstatus', axis=1)
```
• Dropped furnishingstatus: This feature was removed because it did not significantly impact the model's predictive performance.

• Binary encoding: Converted categorical "yes/no" columns into boolean True/False for compatibility with ML algorithms.
```python
cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[cols] = df[cols].replace({'yes': True, 'no': False})
```

##🏐 Feature Engineering

A new feature total_rooms was created by summing bedrooms and bathrooms to better capture the overall size of the property:
```python
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
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
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
```

---

## 🧐 Model Building & Evaluation

We used GradientBoostingRegressor from sklearn.ensemble, with hyperparameter tuning via GridSearchCV.
```python
params = [{
    'n_estimators': [30, 50, 70],
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [2, 5]
}]
```
• Data split: 80% training / 20% testing

• Evaluation metric: R² Score (coefficient of determination)
```python
print(f'R2: {r2_score(y_test, y_pred)}')
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

• Google Colab as the development environment

---

## 📬 Author

Telegram: @lesin_official

Email: dmitrylesin_official@gmail.com

© 2025 Dmitry Lesin. All rights reserved.

