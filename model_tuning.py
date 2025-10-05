# model_tuning.py
import os, json
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb

from config import ARTIFACTS_DIR, GRID
from utils_split import chrono_split

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def main():
    X = pd.read_csv(f"{ARTIFACTS_DIR}/X.csv")
    y = pd.read_csv(f"{ARTIFACTS_DIR}/y.csv")["y"]
    ts = pd.read_csv(f"{ARTIFACTS_DIR}/timestamps.csv")["hour_timestamp"]

    X_tr, y_tr, X_val, y_val, X_te, y_te, _ = chrono_split(X, y, ts)

    tscv = TimeSeriesSplit(n_splits=3)
    model = xgb.XGBRegressor(random_state=42, n_jobs=-1, tree_method="hist")

    gcv = GridSearchCV(
        estimator=model,
        param_grid=GRID,
        scoring="neg_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
        verbose=1
    )
    gcv.fit(X_tr, y_tr)
    best = gcv.best_estimator_
    print("Best params:", gcv.best_params_)

    best.save_model(f"{ARTIFACTS_DIR}/xgb_model.json")
    with open(f"{ARTIFACTS_DIR}/best_params.json", "w") as f:
        json.dump(gcv.best_params_, f, indent=2)

    print("Saved: xgb_model.json, best_params.json")

if __name__ == "__main__":
    main()
