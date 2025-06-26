import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from collections import Counter

import logging

logging.basicConfig(
  level=logging.INFO,
  filename='log.txt',
  filemodel='a',
  format='%(astime)s - %(levelname)s - %(message)s'
)

logging.info('Dataset loaded successfully.')
df = pd.read_csv('hate_speech_1829.csv')

df.head()
df.info()
df.isnull().sum()

df = df.drop(['post_id', 'user_id', 'language', 'preprocessed_text'], axis=1)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['month'] = df['timestamp'].dt.month
df['year'] = df['timestamp'].dt.year

df = df.drop(columns=['timestamp'])
le_label = LabelEncoder()
y_encoder = le_label.fit_transform(df['label'])
