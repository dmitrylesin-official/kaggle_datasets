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

