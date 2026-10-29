"""가중 평균 앙상블. 가중치는 검증 구간 예측에 대한 비음수 최소제곱(NNLS)으로 정하고 합을 1로 맞춘다."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def fit_weights(preds: dict[str, np.ndarray], y: np.ndarray) -> dict[str, float]:
    names = list(preds)
    A = np.column_stack([preds[n] for n in names])
    w, _ = nnls(A, np.asarray(y, float))
    if w.sum() == 0:
        w = np.ones(len(names))
    w = w / w.sum()
    return {n: float(v) for n, v in zip(names, w)}


def blend(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(weights[n] * np.asarray(preds[n], float) for n in weights)
