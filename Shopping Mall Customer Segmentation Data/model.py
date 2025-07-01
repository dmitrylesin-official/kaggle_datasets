import pandas as pd
import logging

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

# Configure logging to capture important business insights and errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='clustering.log',
    filemode='w'
)

# Load dataset
try:
    df = pd.read_csv('Shopping Mall Customer Segmentation Data .csv', encoding='utf-8')
    logging.info(f'Dataset successfully loaded. Shape: {df.shape}')
except Exception as e:
    logging.error(f'Error loading dataset: {e}')
    raise

# One-hot encode gender column (drop first to avoid multicollinearity)
encoder = OneHotEncoder(sparse_output=False, drop='first')
gender_encoded = encoder.fit_transform(df[['Gender']])
gender_df = pd.DataFrame(gender_encoded, columns=encoder.get_feature_names_out(['Gender']))

# Merge encoded features and drop irrelevant columns
df_encoded = pd.concat([df.drop(['Gender', 'CustomerID'], axis=1), gender_df], axis=1)

# Standardize numeric features (scale age, income, and spending)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)
X_scaled = pd.DataFrame(X_scaled, columns=df_encoded.columns)

# Elbow method to determine optimal number of clusters
inertia = []
K_range = range(1, 21)

logging.info('Running elbow method for KMeans clustering...')
for k in K_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertia.append(model.inertia_)
    logging.info(f'k={k}, Inertia={model.inertia_:.2f}')

# Plot the elbow curve
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid()
plt.tight_layout()
plt.savefig('elbow_plot.png')
plt.show()

# Apply KMeans with chosen number of clusters (business can adjust k based on elbow plot)
k_final = 10
kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df_encoded['Cluster'] = clusters

# Log the number of customers in each cluster
cluster_counts = df_encoded['Cluster'].value_counts().to_dict()
logging.info(f'Customer distribution per cluster (k={k_final}): {cluster_counts}')

# Get average profile of each customer segment — useful for marketing/business strategy
cluster_profile = df_encoded.groupby('Cluster').mean()
logging.info(f'Average feature values per cluster:\n{cluster_profile}')

# Evaluate clustering quality using silhouette score
score = silhouette_score(X_scaled, df_encoded['Cluster'])
logging.info(f'Silhouette score for k={k_final}: {score:.3f}')
print(f'Silhouette Score: {score:.3f}')
