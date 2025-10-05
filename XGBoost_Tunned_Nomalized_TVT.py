import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


# Load and preprocess dataset
df = pd.read_csv("/Users/khashayar/Desktop/BDSD Minor Project/Electrical data/device1_phases_with_day_and_temp.csv")
df['reading_date'] = pd.to_datetime(df['reading_date'])
df['hour'] = df['reading_date'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Scale temperature only
scaler = StandardScaler()
df['temperature_scaled'] = scaler.fit_transform(df[['temperature']])

# Select features and target
features = ['temperature_scaled', 'hour_sin', 'hour_cos', 'day_of_week']
df_model = df[features + ['total_power']]
df_encoded = pd.get_dummies(df_model, columns=['day_of_week'], drop_first=True)

X = df_encoded.drop(columns=['total_power'])
y = df_encoded['total_power']

# Split into train (64%), val (16%), test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    colsample_bytree=1.0,
    reg_alpha=0,
    reg_lambda=1,
    subsample=1.0,
    random_state=42
)
model.fit(X_train, y_train)

# Predict and evaluate on all three sets
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

rmse_train = sqrt(mean_squared_error(y_train, y_train_pred))
rmse_val = sqrt(mean_squared_error(y_val, y_val_pred))
rmse_test = sqrt(mean_squared_error(y_test, y_test_pred))

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Train RMSE:", rmse_train, "| R²:", r2_train)
print("Val   RMSE:", rmse_val, "| R²:", r2_val)
print("Test  RMSE:", rmse_test, "| R²:", r2_test)

# Plot predicted vs actual on validation
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:200], label='Actual', linewidth=2)
plt.plot(y_test_pred[:200], label='Predicted', linestyle='--')
plt.title("Predicted vs Actual Total Power (Test Set, First 200 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Total Power")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Metrics
mae = mean_absolute_error(y_test, y_test_pred)
mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100  # in %
print("MAE:", mae)
print("MAPE:", mape, "%")

# Residuals
residuals = y_test - y_test_pred

# Residual Plot
plt.figure(figsize=(10, 4))
plt.scatter(y_test_pred, residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Predicted")
plt.xlabel("Predicted Total Power")
plt.ylabel("Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()

# Distribution of residuals
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=50, alpha=0.7)
plt.title("Distribution of Residuals")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot predicted vs actual on validation
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual', linewidth=2)
plt.plot(y_test_pred, label='Predicted', linestyle='--')
plt.title("Predicted vs Actual Total Power (Test Set, complete two weeks)")
plt.xlabel("Sample Index")
plt.ylabel("Total Power")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()