# 📞 Customer Churn Analysis & Prediction

## 📥 You can download the dataset here:
🔗 [Churn BigML Dataset](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets) (CSV)

This project focuses on analyzing and predicting customer churn using a real telecom dataset. We perform data cleaning, transformation, visualization, and apply machine learning models — primarily Random Forest — to detect potential customer churn.

---

## 📂 Dataset
The dataset is loaded from a CSV file named churn-bigml-20.csv, which contains telecom service usage and customer data, including:

• Call durations and charges (day, evening, night, international)

• Whether the user has an international plan or voicemail plan

• Customer service call frequency

• Binary target variable: Churn

---

## 🧹 Data Preprocessing
```python
df = pd.read_csv('/content/churn-bigml-20.csv')
df = df.drop('State', axis=1)
```
• Dropped the State column: It does not provide useful signal for churn classification.

• Converted binary yes/no fields into boolean for ML model compatibility:
```python
df['International plan'] = df['International plan'].map({'Yes': True, 'No': False})
df['Voice mail plan'] = df['Voice mail plan'].map({'Yes': True, 'No': False})
```

---

## 📊 Data Visualization
Correlation Heatmap
To identify relationships between numeric features and target variable:
```python
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```
![image](https://github.com/user-attachments/assets/5955728e-d11a-4f83-b1bb-df06a3fd88d1)

This helped in selecting which features could be dropped due to low correlation with the target.

---

## 🗑 Feature Reduction
We manually dropped features that had low correlation or were redundant:
```python
df = df.drop(['Total intl minutes', 'Total night calls', 'Total eve charge',
              'Total day charge', 'Number vmail messages', 'Total night charge'], axis=1)
```

---

## 🧠 Feature Selection (Optional Step)
We initially applied **SelectKBest** with **f_classif** for feature selection:
```python
from sklearn.feature_selection import SelectKBest, f_classif
SelectKBest(score_func=f_classif, k=10)
```
A basic train_test_split was used:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```
We also tested GridSearchCV for hyperparameter tuning:
```python
params = {'max_depth': [10, 15, 20], 'n_estimators': [100, 200], 'min_samples_split': [2, 5, 8]}
grid = GridSearchCV(RandomForestClassifier(), params, cv=5)
```
But in this case, the tuned model performed worse than our initial baseline — likely due to small dataset size and already good baseline parameters.

---

## 📈 Final Results
```
Accuracy: 97.76%
F1-score: 88.88%
```
The model performs exceptionally well on the test set, indicating good predictive power for detecting churned customers.

---

## 🛠 Technologies Used
• Python

• Pandas, Matplotlib, Seaborn for data preprocessing and visualization

• Scikit-learn for model building, training, and evaluation

• Google Colab for running the code

---

## 📋 Logging System
This project includes a lightweight logging system to track key pipeline steps and improve observability.

Key logged events include:

• Dataset load success and shape

• Model training and evaluation results

Logs are automatically written to **log.txt** with timestamps and severity levels (**INFO**, **WARNING**, **ERROR**) using Python’s built-in **logging** module:
```python
logging.basicConfig(
    level=logging.INFO,
    filename='log.txt',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```
This enables faster debugging and provides traceability in case of errors or unexpected results.

---

## 📬 Author
Telegram: @dmitrylesin

Email: dmitrylesin_official@gmail.com

© 2025 Dmitry Lesin. All rights reserved.
