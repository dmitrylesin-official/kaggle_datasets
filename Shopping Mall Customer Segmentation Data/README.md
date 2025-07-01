# 🛍️ Customer Segmentation with KMeans (Shopping Mall Dataset)

📥 Dataset source:

🔗 [Kaggle - Mall Customer Segmentation Data](https://www.kaggle.com/datasets/zubairmustafa/shopping-mall-customer-segmentation-data)

This project focuses on segmenting mall customers into meaningful groups based on their demographics and spending behavior using KMeans clustering. The result helps businesses understand their customer base and tailor marketing strategies accordingly.

---

## 📂 Dataset

The dataset contains customer records with the following features:

**• CustomerID** — unique ID (removed later)

**• Gender** — categorical (encoded)

**• Age** — numeric

**• Annual Income (k$)** — numeric

**• Spending Score (1–100)** — numeric

---

## 🧹 Data Preprocessing & Logging
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    filename='clustering.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

**• Gender encoding:** Applied one-hot encoding with drop='first' to avoid multicollinearity.

**• Dropped CustomerID:** Not useful for clustering logic.

**• Standardization:** Used StandardScaler to bring all numeric features to a common scale — essential for KMeans performance.

**• Logging:** Added logs for dataset loading, cluster distribution, and silhouette scores to monitor performance and reproducibility.

---

## 🧠 Why 10 Clusters?
We chose 10 clusters based on the Elbow Method and Silhouette Score. While the elbow curve lacked a clear “knee,” 10 clusters gave more detailed segmentation than 3–5 — helping separate high vs. low spenders across different age and income groups. This level of granularity better matches real marketing needs. Final Silhouette Score ≈ 0.28 — reasonable for this type of data.
```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X_scaled, df_encoded['Cluster'])
print(f'Silhouette_score: {score:.3f}')
```

---

## 📊 Data Visualization
Elbow Method
A line plot showing how inertia (within-cluster variance) decreases with increasing k. Used to roughly estimate a good cluster range.
```python
plt.plot(K_range, inertia, 'bo-')
plt.title('Elbow Method')
```
![image](https://github.com/user-attachments/assets/87de367c-dc37-453f-b786-00dfa6e1f0ed)

---

## 📈 Cluster Profiling
After assigning customers to clusters, we calculated the mean values of features per cluster to interpret them:
```python
df_encoded.groupby('Cluster').mean()
```
This table helps in building customer personas, e.g.:
• Cluster 3 = Young, low income, low spending

• Cluster 6 = Middle-aged, high income, very high spending

• Cluster 9 = Older, moderate income, disengaged

These insights can inform:
• Personalized promotions

• Loyalty programs

• Offline marketing strategy

• Inventory planning

---

## 🧰 Technologies Used
• Python 3

• Pandas, NumPy for data processing

• Matplotlib for visualization

• Scikit-learn (KMeans, preprocessing, silhouette score)

• Logging for experiment tracking

• Google Colab / Jupyter Notebook

---

## 📬 Author
Telegram: @dmitrylesin

Email: dmitrylesin_official@gmail.com

© 2025 Dmitry Lesin. All rights reserved.
