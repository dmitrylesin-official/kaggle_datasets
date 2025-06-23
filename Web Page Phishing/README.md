# 🛡️ Phishing Website Detection using XGBoost

## 📥 Dataset Source:
🔗 [Web Page Phishing Dataset](https://www.kaggle.com/datasets/danielfernandon/web-page-phishing-dataset)

This project focuses on detecting phishing websites using the XGBoost algorithm. We perform data inspection, correlation analysis, model training with class imbalance handling, and evaluate the model using recall.

---

## 📂 Dataset Overview
The dataset is loaded from a CSV file named /content/web-page-phishing.csv.
It contains various numerical features describing web pages, along with a binary target column:

• phishing = 1 → phishing site

• phishing = 0 → legitimate site

---

## 🧹 Data Preprocessing
```python
df = pd.read_csv('/content/web-page-phishing.csv')
df.info()
df.isnull().sum()
```
• Checked for missing values and data types

• Target variable is phishing

• All features are numeric and clean

---

## 📊 Data Visualization
Target Distribution:
```python
df['phishing'].hist()
```
![image](https://github.com/user-attachments/assets/287457ee-d091-43c2-99c6-da77954dc02b)

Correlation Heatmap:
```python
plt.subplots(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()
```
![image](https://github.com/user-attachments/assets/631b83ad-6f64-4c9c-b2e3-af2fb33ae3ca)

This helps identify which features are strongly correlated with the target and with each other

---

## 🧠 Train/Test Split & Class Balancing
```python
X = df.drop('phishing', axis=1)
y = df['phishing']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```
⚖️ Handling Imbalanced Classes:
```python
from collections import Counter
counts = Counter(y_train)
scale_pos_weight = counts[0] / counts[1]
```
We apply **scale_pos_weight** in XGBoost to account for the imbalance — phishing cases are minority class.

---

## 🚀 Model Training
```python
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight
)

model.fit(X_train, y_train)
```
A simple XGBoost classifier with log-loss objective and weight adjustment for skewed target classes.

---

## 📉 CatBoost Learning Curve Visualization
In addition to the main XGBoost pipeline, you can use the following code snippet to analyze the training process of a CatBoost model:
```
results = model.get_evals_result()

plt.figure(figsize=(10, 6))
plt.plot(results['learn']['Logloss'], label='Train')
plt.plot(results['validation']['Logloss'], label='Validation')
plt.xlabel('Iteration')
plt.ylabel('Logloss')
plt.title('CatBoost Learning Curve')
plt.legend()
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/81f216d3-818d-425c-96a4-e86c340d658c)

What this code does:

• model.get_evals_result() returns a dictionary with metric values recorded at each training iteration.

• The plot shows Logloss curves over iterations for both training and validation datasets.

• This visualization helps to see if the model is overfitting, underfitting, or training properly.

• Useful for deciding the number of iterations, applying early stopping, and tuning hyperparameters.

## 📈 Model Evaluation
```python
y_pred = model.predict(X_test)
print(f'Recall: {recall_score(y_test, y_pred)}')
```
**Evaluation Metric:** Recall is prioritized here, as false negatives (missed phishing sites) are more dangerous than false positives.

---

## 🔍 Feature Importance
```python
plot_importance(model, max_num_features=20, importance_type='gain')
plt.show()
```
![image](https://github.com/user-attachments/assets/6753c522-52b9-44ee-a261-96a4f1879e90)
Shows which features contribute most to the model’s predictions, based on gain.

---

## 📋 Logging System
```python
logging.basicConfig(
    level=logging.INFO,
    filename='log.txt',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```
Logging is used to track key events in the pipeline:
```python
logging.INFO('Dataset loaded successfully')
```
All logs are written to **log.txt** with timestamps and severity levels. This helps with debugging and tracing pipeline steps

---

## 🛠 Technologies Used

• Python

• Pandas, Seaborn, Matplotlib for analysis and visualization

• Scikit-learn for data splitting and evaluation

• XGBoost for classification

• Logging module for tracking and observability

---

## 📬 Author
Telegram: @dmitrylesin

Email: dmitrylesin_official@gmail.com

© 2025 Dmitry Lesin. All rights reserved.
