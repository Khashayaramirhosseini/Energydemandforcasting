# config.py
CSV_PATH = "device_data_woningen.csv"  # <- change if needed

TIMESTAMP_COL = "reading_date"
TARGET_COL    = "total_power"
TEMP_COL      = "temperature"

# splits (by hours, chronological)
TEST_HOURS = 24 * 7   # last 7 days test
VAL_HOURS  = 24 * 3   # last 3 days val

# XGB grid
GRID = {
    "n_estimators": [100, 150, 300],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.01, 0.05, 0.10],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda": [1, 5, 10],
    "reg_alpha": [0, 0.1, 1.0]
}

ARTIFACTS_DIR = "artifacts"
