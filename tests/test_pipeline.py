import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.costs import best_threshold, total_cost
from src.data import time_split


def make_df(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"Time": rng.permutation(n), "V1": rng.normal(size=n),
                       "Amount": rng.uniform(1, 200, n)})
    df["Class"] = (rng.random(n) < 0.05).astype(int)
    return df.sort_values("Time").reset_index(drop=True)


def test_split_has_no_time_overlap():
    train, val, test = time_split(make_df())
    assert train["Time"].max() < val["Time"].min()
    assert val["Time"].max() < test["Time"].min()


def test_split_covers_all_rows():
    df = make_df()
    assert sum(len(p) for p in time_split(df)) == len(df)


def test_total_cost_known_example():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]
    amounts = [10, 50, 20, 30]
    # missed fraud = 50, one false alarm = 5
    assert total_cost(y_true, y_pred, amounts, fp_cost=5) == 55


def test_best_threshold_in_range():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.1).astype(int)
    probs = np.clip(y * 0.6 + rng.random(500) * 0.4, 0, 1)
    t, cost = best_threshold(y, probs, np.full(500, 100.0), fp_cost=5)
    assert 0 < t < 1 and cost >= 0


def test_model_outputs_valid_probabilities():
    df = make_df()
    X, y = df[["V1", "Amount"]], df["Class"]
    probs = XGBClassifier(n_estimators=10).fit(X, y).predict_proba(X)[:, 1]
    assert ((probs >= 0) & (probs <= 1)).all()
