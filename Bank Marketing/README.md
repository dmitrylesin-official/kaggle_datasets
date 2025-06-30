# 📞 Bank Marketing Prediction

📥 You can download the dataset from UCI or other sources:

🔗 [Bank Marketing Dataset](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv)

This project focuses on predicting whether a client will subscribe to a term deposit after a direct marketing campaign, using CatBoost — a gradient boosting model optimized for categorical features.

We specifically optimize for Recall, prioritizing capturing as many potential clients as possible. Since phone calls are relatively cheap for banks, missing a potential customer is more costly than calling an uninterested one.

---

## 🎯 Problem Statement
The data is related to direct marketing campaigns (via phone calls) conducted by a Portuguese banking institution.

**Goal:** Predict if a client will subscribe to a term deposit (target variable: y).

**• Target is binary:** 'yes' → client subscribed, 'no' → client did not subscribe

**• Focus on Recall:** better to falsely assume interest than to miss a real lead.

---

## 📂 Dataset
The data is loaded from a CSV file bank-additional-full.csv (separator = ;) and includes:

**• Client info:** age, job, marital status, education, etc.

**• Contact details:** contact type, campaign duration, etc.

**• Economic indicators:** employment variation, consumer confidence, etc.

**• Target variable:** y (term deposit subscription)

---

## 🧹 Data Preprocessing
```python
df = pd.read_csv('bank-additional-full.csv', sep=';')
```
• Dropped "default" column due to high % of unknowns

• Converted 'y' to boolean: yes → True, no → False

• Mapped loan and housing to numeric (except 'unknown' left as string)

• Categorical columns are handled natively by CatBoost
```python
df['y'] = df['y'].replace({'yes': True, 'no': False})
X = df.drop(columns='y')
y = df['y']
```
• 80/20 train-test split

• Categorical columns extracted automatically for CatBoost's Pool

---

## ⚖️ Class Imbalance Handling
Since the target variable is imbalanced, we manually calculated class weights to help the model pay more attention to minority class (y=True):

```python
from collections import Counter

counts = Counter(y_train)
total = sum(counts.values())
class_weights = {
    0: total / (2 * counts[0]),
    1: total / (2 * counts[1]),
}
```

---

## 🚀 Model Training with CatBoost
We use CatBoostClassifier, which handles categorical data natively and performs well with tabular datasets:
```python
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=8,
    eval_metric='Recall',
    random_state=42,
    verbose=100,
    class_weights=class_weights
)
```
• Evaluation set: test_pool

• Early stopping enabled to prevent overfitting
```python
model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)
```

---

## 📊 Evaluation: Focus on Recall
After training, we evaluate the model using classic classification metrics:
```python
from sklearn.metrics import precision_score, recall_score, f1_score

y_pred = model.predict(test_pool)
print("Recall:", recall_score(y_test, y_pred))
```
📌 **Example output:**
```
Recall: 0.9401
```
This high recall means the model correctly captures ~94% of actual subscribers — which is crucial for marketing campaigns, where missing a potential lead can be costly.

---

## 🛠 Technologies Used
• Python

• Pandas — data preprocessing

• CatBoost — classification model

• Scikit-learn (sklearn) — train-test split and evaluation

• Google Colab — for development and training

---

## 📬 Author
Telegram: @dmitrylesin
Email: dmitrylesin_official@gmail.com

© 2025 Dmitry Lesin. All rights reserved.
