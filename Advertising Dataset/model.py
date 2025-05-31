import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('advertising.csv')
df.head()
df.info()

X = df.drop('Sales', axis=1)
y = df['Sales']

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(random_state=42)

params = [{
    'n_estimators': [30, 50, 70],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4]
}]

grid_search = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=5,
    scoring='r2'
)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(abs(grid_search.best_score_))

y_pred = grid_search.predict(X_test)

print(f'MSE: {mean_squared_error(y_test, y_pred)}')
print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
