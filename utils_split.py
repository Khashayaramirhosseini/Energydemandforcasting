# utils_split.py
import pandas as pd
from config import TEST_HOURS, VAL_HOURS

def chrono_split(X: pd.DataFrame, y: pd.Series, ts: pd.Series):
    n = len(ts)
    test_start = max(0, n - TEST_HOURS)
    val_start  = max(0, test_start - VAL_HOURS)

    X_train = X.iloc[:val_start].values
    y_train = y.iloc[:val_start].values
    X_val   = X.iloc[val_start:test_start].values
    y_val   = y.iloc[val_start:test_start].values
    X_test  = X.iloc[test_start:].values
    y_test  = y.iloc[test_start:].values
    ts_test = ts.iloc[test_start:].reset_index(drop=True)

    return X_train, y_train, X_val, y_val, X_test, y_test, ts_test
