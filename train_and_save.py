# train_and_save.py
import os, json
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from config import ARTIFACTS_DIR
from utils_split import chrono_split

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def report(name, y, yp):
    rmse = np.sqrt(mean_squared_error(y, yp))
    r2   = r2_score(y, yp)
    mae  = mean_absolute_error(y, yp)
    print(f"{name:<5} | RMSE: {rmse:.2f} | R²: {r2:.3f} | MAE: {mae:.2f}")
    return {"rmse": float(rmse), "r2": float(r2), "mae": float(mae)}

def main():
    X = pd.read_csv(f"{ARTIFACTS_DIR}/X.csv")
    y = pd.read_csv(f"{ARTIFACTS_DIR}/y.csv")["y"]
    ts = pd.read_csv(f"{ARTIFACTS_DIR}/timestamps.csv")["hour_timestamp"]

    X_tr, y_tr, X_val, y_val, X_te, y_te, ts_test = chrono_split(X, y, ts)

    model = xgb.XGBRegressor()
    model.load_model(f"{ARTIFACTS_DIR}/xgb_model.json")

    X_final = np.vstack([X_tr, X_val])
    y_final = np.hstack([y_tr, y_val])
    model.fit(X_final, y_final)

    y_trp = model.predict(X_tr)
    y_valp = model.predict(X_val)
    y_tep = model.predict(X_te)

    m_tr  = report("Train", y_tr, y_trp)
    m_val = report("Val",   y_val, y_valp)
    m_te  = report("Test",  y_te,  y_tep)

    pd.DataFrame({
        "hour_timestamp": ts_test.values,
        "y_true": y_te,
        "y_pred": y_tep
    }).to_csv(f"{ARTIFACTS_DIR}/predictions_test.csv", index=False)

    with open(f"{ARTIFACTS_DIR}/metrics.json", "w") as f:
        json.dump({"train": m_tr, "val": m_val, "test": m_te}, f, indent=2)

    model.save_model(f"{ARTIFACTS_DIR}/xgb_model_final.json")
    print("Saved: predictions_test.csv, metrics.json, xgb_model_final.json")

if __name__ == "__main__":
    main()
