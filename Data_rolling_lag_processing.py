import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

# Train XGBoost model
model = xgb.XGBRegressor(
    colsample_bytree=0.6,
    learning_rate=0.05,
    max_depth=3,
    n_estimators=50,
    reg_alpha=0.1,
    reg_lambda=5,
    subsample=0.6,
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Evaluation function
def evaluate(label, y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{label} | RMSE: {rmse:.2f} | R²: {r2:.3f} | MAE: {mae:.2f}")

# Print evaluations
evaluate("Train", y_train, y_train_pred)
evaluate("Val  ", y_val, y_val_pred)
evaluate("Test ", y_test, y_test_pred)

# Plot predicted vs actual for validation
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label='Actual', linewidth=2)
plt.plot(y_test_pred[:100], label='Predicted', linestyle='--')
plt.title("XGBoost with Lag & Rolling Features - Validation Set")
plt.xlabel("Hour Index")
plt.ylabel("Total Power")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot top N features by importance (default = gain)
xgb.plot_importance(model, importance_type='gain', max_num_features=10, height=0.5)
plt.title("Top Feature Importances (Gain)")
plt.tight_layout()
plt.show()