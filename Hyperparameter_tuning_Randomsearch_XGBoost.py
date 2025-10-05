from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from math import sqrt


# Reload and preprocess the original dataset
df = pd.read_csv("/Users/khashayar/Desktop/BDSD Minor Project/Electrical data/device1_phases_with_day_and_temp.csv")
df['reading_date'] = pd.to_datetime(df['reading_date'])
df['hour'] = df['reading_date'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Normalize only the external feature: temperature
scaler = StandardScaler()
df['temperature_scaled'] = scaler.fit_transform(df[['temperature']])

# Create the clean feature set (exclude phase_1/2/3 to prevent leakage)
features = ['temperature_scaled', 'hour_sin', 'hour_cos', 'day_of_week']
df_model = df[features + ['total_power']]

# One-hot encode day_of_week
df_encoded = pd.get_dummies(df_model, columns=['day_of_week'], drop_first=True)

# Split into features and target
X = df_encoded.drop(columns=['total_power'])
y = df_encoded['total_power']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a broader range of hyperparameters for randomized search
param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Initialize model and RandomizedSearchCV
xgb_model = xgb.XGBRegressor(random_state=42)
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    scoring='neg_mean_squared_error',
    n_iter=10,  # limit for speed
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=1
)

# Fit and evaluate
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Final evaluation
rmse_best = sqrt(mean_squared_error(y_test, y_pred_best))
r2_best = r2_score(y_test, y_pred_best)

print(random_search.best_params_, rmse_best, r2_best)
