# Re-import after kernel reset and rerun the corrected preprocessing + training script
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from math import sqrt
import matplotlib.pyplot as plt


# Reload and preprocess the original dataset
df = pd.read_csv("/Users/khashayar/Desktop/BDSD Minor Project/Electrical data/device1_phases_with_day_and_temp.csv")
df['reading_date'] = pd.to_datetime(df['reading_date'])
df['hour'] = df['reading_date'].dt.hour
print(df['hour'])
print(np.size(df['hour']))
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Normalize only the external feature: temperature
scaler = StandardScaler()
df['temperature_scaled'] = scaler.fit_transform(df[['temperature']])

# Create the clean feature set (exclude phase_1/2/3 to prevent leakage)
features = ['temperature_scaled', 'hour_sin', 'hour_cos', 'day_of_week']
df_model = df[features + ['total_power']]
# Save df_model to Excel file
df_model.to_excel("/Users/khashayar/Desktop/BDSD Minor Project/Electrical data/df_model.xlsx", index=False)

# One-hot encode day_of_week
df_encoded = pd.get_dummies(df_model, columns=['day_of_week'], drop_first=True)

# Split into features and target
X = df_encoded.drop(columns=['total_power'])
y = df_encoded['total_power']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(rmse, r2)


# Plot predicted vs actual total power
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:200], label='Actual', linewidth=2)
plt.plot(y_pred[:200], label='Predicted', linestyle='--')
plt.title("Predicted vs Actual Total Power (First 200 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Total Power")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()