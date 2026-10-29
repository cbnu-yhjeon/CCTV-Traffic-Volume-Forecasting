"""지연값 특징. 예측 시점(t - horizon)에 이미 관측된 값만 쓴다.

기본 horizon=1D: 전날까지의 집계로 다음 날 시간대별 교통량을 예측하는 설정.
시계열을 고정 간격으로 다시 맞춘 뒤 '행 수'가 아니라 '시간'만큼 민다(결측 구간이 있어도 안전).
"""

from __future__ import annotations

import pandas as pd

LAG_DAYS = [1, 2, 7, 14]


def lag_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("lag_", "roll", "same_slot"))]


def add_lag_features(df: pd.DataFrame, target: str = "volume", freq: str = "15min",
                     horizon: str = "1D") -> pd.DataFrame:
    step = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    h = pd.Timedelta(horizon)
    full = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    y = df[target].reindex(full)
    shift = lambda delta: y.shift(int(delta / step))

    feats = pd.DataFrame(index=full)
    for d in LAG_DAYS:
        if pd.Timedelta(days=d) >= h:
            feats[f"lag_{d}d"] = shift(pd.Timedelta(days=d))
    # 같은 요일 · 같은 시각의 최근 4주 평균(계절성 기준선을 부드럽게 만든 값)
    weekly = [shift(pd.Timedelta(days=7 * k)) for k in range(1, 5) if pd.Timedelta(days=7 * k) >= h]
    feats["same_slot_4w_mean"] = pd.concat(weekly, axis=1).mean(axis=1, skipna=True)
    # 예측 시점 직전 24시간의 수준 · 변동
    base = shift(h)
    window = int(pd.Timedelta("1D") / step)
    feats["roll24h_mean"] = base.rolling(window, min_periods=window // 2).mean()
    feats["roll24h_std"] = base.rolling(window, min_periods=window // 2).std()
    feats["roll7d_mean"] = base.rolling(window * 7, min_periods=window * 3).mean()

    return df.join(feats, how="left")
