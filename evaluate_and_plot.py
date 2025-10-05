# evaluate_and_plot.py
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from config import ARTIFACTS_DIR

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def main():
    preds = pd.read_csv(f"{ARTIFACTS_DIR}/predictions_test.csv")
    fnames = json.load(open(f"{ARTIFACTS_DIR}/feature_names.json"))
    X = pd.read_csv(f"{ARTIFACTS_DIR}/X.csv")

    model = xgb.XGBRegressor()
    model.load_model(f"{ARTIFACTS_DIR}/xgb_model_final.json")

    m = min(len(preds), 200)
    plt.figure(figsize=(10,4))
    plt.plot(preds["y_true"].values[:m], label="Actual", linewidth=2)
    plt.plot(preds["y_pred"].values[:m], label="Pred", linestyle="--")
    plt.title("Test — first 200 hours")
    plt.xlabel("Hour index"); plt.ylabel("Total Power")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(f"{ARTIFACTS_DIR}/test_first200.png", dpi=150)
    plt.show()

    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    pairs = []
    for k, v in score.items():
        try:
            idx = int(k[1:])
            fname = fnames[idx] if idx < len(fnames) else k
        except:
            fname = k
        pairs.append((fname, v))
    pairs.sort(key=lambda t: t[1], reverse=True)
    top = pairs[:20]

    plt.figure(figsize=(10,6))
    labels = [p[0] for p in top][::-1]
    gains  = [p[1] for p in top][::-1]
    plt.barh(labels, gains)
    plt.title("Top Feature Importances (gain)")
    plt.tight_layout()
    plt.savefig(f"{ARTIFACTS_DIR}/feature_importance.png", dpi=150)
    plt.show()

    print("Saved: test_first200.png, feature_importance.png")

if __name__ == "__main__":
    main()
