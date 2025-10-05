# Apply proper cyclical transformation for hour and re-normalize other features
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Reload dataset for safety
df = pd.read_csv("/Users/khashayar/Desktop/BDSD Minor Project/Electrical data/device1_phases_with_day_and_temp.csv")
df['reading_date'] = pd.to_datetime(df['reading_date'])
df['hour'] = df['reading_date'].dt.hour

# Add cyclical hour features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Normalize only temperature and power phases (not hour anymore)
features_to_scale = ['temperature']
scaler = StandardScaler()
scaled_values = scaler.fit_transform(df[features_to_scale])
df_scaled = pd.DataFrame(scaled_values, columns=features_to_scale)

# Add cyclical and categorical features
df_scaled['hour_sin'] = df['hour_sin']
df_scaled['hour_cos'] = df['hour_cos']
df_scaled['day_of_week'] = df['day_of_week']

# Visualize updated power-related features with boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_scaled[['temperature']])
plt.title("Boxplot of Normalized Power Phases and Total Power (with Cyclical Time)")
plt.grid(True)
plt.tight_layout()
plt.show()
