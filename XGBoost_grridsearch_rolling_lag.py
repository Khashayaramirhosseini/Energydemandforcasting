import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from math import sqrt
import matplotlib.pyplot as plt

# Reload the uploaded dataset and apply lag and rolling features
df = pd.read_csv("/Users/khashayar/Desktop/BDSD Minor Project/Electrical data/device1_phases_with_day_and_temp.csv")
df['reading_date'] = pd.to_datetime(df['reading_date'])

# Round timestamps to the nearest hour and aggregate by hour
df['hour_timestamp'] = df['reading_date'].dt.floor('h')
df_hourly = df.groupby('hour_timestamp').agg({
    'temperature': 'mean',
    'phase_1': 'mean',
    'phase_2': 'mean',
    'phase_3': 'mean',
    'total_power': 'mean'
}).reset_index()

# Add time-based features
df_hourly['hour'] = df_hourly['hour_timestamp'].dt.hour
df_hourly['day_of_week'] = df_hourly['hour_timestamp'].dt.day_name()
df_hourly['hour_sin'] = np.sin(2 * np.pi * df_hourly['hour'] / 24)
df_hourly['hour_cos'] = np.cos(2 * np.pi * df_hourly['hour'] / 24)

# Rolling window features (past 3 hours)
df_hourly['power_mean_last_3h'] = df_hourly['total_power'].rolling(window=3).mean().shift(1)
df_hourly['power_std_last_3h'] = df_hourly['total_power'].rolling(window=3).std().shift(1)
df_hourly['temp_mean_last_3h'] = df_hourly['temperature'].rolling(window=3).mean().shift(1)

# Lag features
df_hourly['power_lag_1h'] = df_hourly['total_power'].shift(1)
df_hourly['power_lag_2h'] = df_hourly['total_power'].shift(2)
df_hourly['temp_lag_1h'] = df_hourly['temperature'].shift(1)

# Drop rows with NaNs due to shifting
df_hourly.dropna(inplace=True)

# Show preview of enriched dataset
print(df_hourly.head())

# One-hot encode day_of_week
df_final = pd.get_dummies(df_hourly, columns=['day_of_week'], drop_first=True)

# Feature set: all engineered features + cyclical time + temperature
features = [
    'temperature', 'hour_sin', 'hour_cos',
    'power_mean_last_3h', 'power_std_last_3h', 'temp_mean_last_3h',
    'power_lag_1h', 'power_lag_2h', 'temp_lag_1h'
] + [col for col in df_final.columns if col.startswith('day_of_week_')]

X = df_final[features]
y = df_final['total_power']

# Split into train, validation, test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# --- STEP 3: Grid search setup ---
param_grid = {
    'n_estimators': [30, 50],
    'max_depth': [2, 3],  # very shallow trees
    'learning_rate': [0.01, 0.05],
    'subsample': [0.6],  # strong regularization
    'colsample_bytree': [0.6],
    'reg_lambda': [5, 10],  # strong L2 regularization
    'reg_alpha': [0.1, 1.0]  # L1 regularization
}



xgb_model = xgb.XGBRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2,
    n_jobs=-1
)

# --- STEP 4: Fit the model ---
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# --- STEP 5: Evaluate performance ---
y_train_pred = best_model.predict(X_train)
y_val_pred = best_model.predict(X_val)
y_test_pred = best_model.predict(X_test)

def evaluate(name, y_true, y_pred):
    print(f"{name} | RMSE: {sqrt(mean_squared_error(y_true, y_pred)):.2f} | R²: {r2_score(y_true, y_pred):.3f} | MAE: {mean_absolute_error(y_true, y_pred):.2f}")

print("\nBest Parameters:", grid_search.best_params_)
evaluate("Train", y_train, y_train_pred)
evaluate("Val  ", y_val, y_val_pred)
evaluate("Test ", y_test, y_test_pred)

# --- STEP 6: Visualization ---
plt.figure(figsize=(10, 5))
plt.plot(y_val.values[:100], label='Actual', linewidth=2)
plt.plot(y_val_pred[:100], label='Predicted', linestyle='--')
plt.title("XGBoost GridSearch - Predicted vs Actual (Validation Set, First 100 Hours)")
plt.xlabel("Hour Index")
plt.ylabel("Total Power (kW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

