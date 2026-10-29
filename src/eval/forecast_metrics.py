"""예측 지표: MAE, RMSE, MAPE, WAPE.

MAPE는 실제값이 0에 가까운 구간(심야 · 장비 오류)에서 폭발하므로 mape_floor 미만은 뺀다.
WAPE(= Σ|오차| / Σ실제값)는 그런 문제가 없어 함께 본다.
"""

from __future__ import annotations

import numpy as np


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mape(y, p, floor: float = 1.0) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y >= floor
    return float(np.mean(np.abs(y[m] - p[m]) / y[m]) * 100) if m.any() else float("nan")


def wape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / np.abs(y).sum() * 100)


def summarize(y, p, mape_floor: float = 1.0) -> dict[str, float]:
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE": mape(y, p, mape_floor), "WAPE": wape(y, p)}
