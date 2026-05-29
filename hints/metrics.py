from __future__ import annotations

import numpy as np
from lifelines.utils import concordance_index
from numba import jit


@jit(nopython=True)
def auc_score(hazard: np.ndarray, status: np.ndarray, time: np.ndarray) -> float:
    total = correct = 0
    n = len(time)
    for i in range(n):
        if status[i] and (time[i] != time.max()) and (time[i] != time.min()):
            for j in range(n):
                if status[j] and (time[j] < time[i]):
                    for k in range(n):
                        if time[k] > time[i]:
                            total += 1
                            correct += hazard[j] > hazard[k]
    return correct / total if total else 0.0


def c_index(event_times, predicted_scores, event_observed=None) -> float:
    try:
        return concordance_index(event_times, predicted_scores, event_observed)
    except ZeroDivisionError:
        return float("nan")
