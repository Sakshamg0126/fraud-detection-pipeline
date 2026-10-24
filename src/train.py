import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import yaml
from sklearn.metrics import average_precision_score, precision_score, recall_score
from xgboost import XGBClassifier

from src.costs import best_threshold, total_cost
from src.data import load_data, time_split


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    target = cfg["data"]["target"]
    time_col = cfg["data"]["time_col"]
    amount_col = cfg["data"]["amount_col"]
    fp_cost = cfg["costs"]["false_positive"]

    df = load_data(cfg["data"]["path"], time_col)
    train, val, test = time_split(df, cfg["split"]["train_frac"], cfg["split"]["val_frac"])

    # Time is dropped as a feature: it only marks "when", and the test set is in the future
    features = [c for c in df.columns if c not in (target, time_col)]
    X_train, y_train = train[features], train[target]
    X_val, y_val = val[features], val[target]
    X_test, y_test = test[features], test[target]

    params = dict(cfg["model"])
    params["scale_pos_weight"] = float((y_train == 0).sum() / (y_train == 1).sum())

    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run():
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        # 1) Choose the threshold on VALIDATION data
        val_probs = model.predict_proba(X_val)[:, 1]
        threshold, _ = best_threshold(y_val, val_probs, val[amount_col], fp_cost)

        # 2) Report on TEST data using that fixed threshold
        test_probs = model.predict_proba(X_test)[:, 1]
        test_pred = test_probs >= threshold

        model_cost = total_cost(y_test, test_pred, test[amount_col], fp_cost)
        no_model_cost = total_cost(y_test, np.zeros(len(y_test)), test[amount_col], fp_cost)

        metrics = {
            "pr_auc": average_precision_score(y_test, test_probs),
            "precision": precision_score(y_test, test_pred, zero_division=0),
            "recall": recall_score(y_test, test_pred),
            "threshold": threshold,
            "test_cost": model_cost,
            "baseline_cost_never_flag": no_model_cost,
            "cost_saved": no_model_cost - model_cost,
        }
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, cfg["paths"]["model"])
        with open(cfg["paths"]["metadata"], "w") as f:
            json.dump({"features": features, "threshold": threshold}, f, indent=2)

        for k, v in metrics.items():
            print(f"{k:>26}: {v:,.3f}")


if __name__ == "__main__":
    main()