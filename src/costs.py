import numpy as np


def total_cost(y_true, y_pred, amounts, fp_cost):
    """Missed fraud costs its Amount; a false alarm costs fp_cost."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred).astype(int)
    amounts = np.asarray(amounts)
    missed = (y_true == 1) & (y_pred == 0)
    false_alarm = (y_true == 0) & (y_pred == 1)
    return float(amounts[missed].sum() + false_alarm.sum() * fp_cost)


def best_threshold(y_true, probs, amounts, fp_cost):
    """Return (threshold, cost) with the lowest total cost."""
    probs = np.asarray(probs)
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = [total_cost(y_true, probs >= t, amounts, fp_cost) for t in thresholds]
    i = int(np.argmin(costs))
    return float(thresholds[i]), float(costs[i])