# feature_engineering.py
import os, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from config import TEMP_COL, TARGET_COL, ARTIFACTS_DIR

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def main():
    df = pd.read_csv(f"{ARTIFACTS_DIR}/hourly.csv")
    df["hour_timestamp"] = pd.to_datetime(df["hour_timestamp"])

    # time/cyclical
    df["hour"] = df["hour_timestamp"].dt.hour
    df["day_of_week"] = df["hour_timestamp"].dt.day_name()
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

    # lag & rolling (shifted)
    df["power_mean_last_3h"] = df[TARGET_COL].rolling(3).mean().shift(1)
    df["power_std_last_3h"]  = df[TARGET_COL].rolling(3).std().shift(1)
    df["temp_mean_last_3h"]  = df[TEMP_COL].rolling(3).mean().shift(1)
    df["power_lag_1h"] = df[TARGET_COL].shift(1)
    df["power_lag_2h"] = df[TARGET_COL].shift(2)
    df["temp_lag_1h"]  = df[TEMP_COL].shift(1)

    df = df.dropna().reset_index(drop=True)

    # one-hot day_of_week
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    dow = ohe.fit_transform(df[["day_of_week"]])
    dow_cols = ohe.get_feature_names_out(["day_of_week"])

    X = pd.DataFrame({
        TEMP_COL: df[TEMP_COL].values,
        "hour_sin": df["hour_sin"].values,
        "hour_cos": df["hour_cos"].values,
        "power_mean_last_3h": df["power_mean_last_3h"].values,
        "power_std_last_3h": df["power_std_last_3h"].values,
        "temp_mean_last_3h": df["temp_mean_last_3h"].values,
        "power_lag_1h": df["power_lag_1h"].values,
        "power_lag_2h": df["power_lag_2h"].values,
        "temp_lag_1h": df["temp_lag_1h"].values,
    })
    X = pd.concat([X, pd.DataFrame(dow, columns=dow_cols, index=X.index)], axis=1)

    y = df[TARGET_COL].values
    feature_names = list(X.columns)

    # save artifacts
    X.to_csv(f"{ARTIFACTS_DIR}/X.csv", index=False)
    pd.Series(y).to_csv(f"{ARTIFACTS_DIR}/y.csv", index=False, header=["y"])
    df[["hour_timestamp"]].to_csv(f"{ARTIFACTS_DIR}/timestamps.csv", index=False)
    with open(f"{ARTIFACTS_DIR}/feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(f"{ARTIFACTS_DIR}/ohe_categories.json", "w") as f:
        json.dump(list(dow_cols), f, indent=2)

    print("Saved: X.csv, y.csv, timestamps.csv, feature_names.json, ohe_categories.json")

if __name__ == "__main__":
    main()
