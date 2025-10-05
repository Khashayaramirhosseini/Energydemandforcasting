# data_preprocessing.py
import os
import pandas as pd
from config import CSV_PATH, TIMESTAMP_COL, TARGET_COL, TEMP_COL, ARTIFACTS_DIR

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def main():
    df = pd.read_csv(CSV_PATH)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COL, TARGET_COL, TEMP_COL])

    df["hour_timestamp"] = df[TIMESTAMP_COL].dt.floor("H")
    df_hourly = df.groupby("hour_timestamp", as_index=False).agg({
        TEMP_COL: "mean",
        TARGET_COL: "mean"
    }).dropna().reset_index(drop=True)

    df_hourly.to_csv(f"{ARTIFACTS_DIR}/hourly.csv", index=False)
    print(f"Saved: {ARTIFACTS_DIR}/hourly.csv")

if __name__ == "__main__":
    main()
