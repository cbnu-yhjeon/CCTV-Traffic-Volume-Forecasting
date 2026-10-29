import numpy as np
import pandas as pd

from src.eval.forecast_metrics import summarize
from src.features.calendar import add_calendar_features
from src.features.lags import add_lag_features, lag_columns


def test_calendar_marks_korean_holidays_and_weekends():
    idx = pd.date_range("2026-09-24", "2026-09-28 23:00", freq="h")  # 2026 추석: 9/24~9/26
    df = add_calendar_features(pd.DataFrame(index=idx), country="KR")
    day = df.resample("D").first()
    assert list(day["is_holiday"]) == [1, 1, 1, 0, 0]
    assert list(day["is_weekend"]) == [0, 0, 1, 1, 0]  # 9/26 토, 9/27 일
    assert list(day["is_offday"]) == [1, 1, 1, 1, 0]
    assert df["hour"].max() == 23
    assert np.allclose(df["hour_sin"] ** 2 + df["hour_cos"] ** 2, 1)


def test_calendar_slot_for_15min():
    idx = pd.date_range("2026-10-29 08:00", periods=4, freq="15min")
    df = add_calendar_features(pd.DataFrame(index=idx), country="KR")
    assert list(df["slot"]) == [32, 33, 34, 35]


def test_lags_use_only_information_older_than_horizon():
    idx = pd.date_range("2026-01-01", periods=24 * 40, freq="h")
    y = pd.Series(np.arange(len(idx), dtype=float), index=idx, name="volume")
    df = add_lag_features(y.to_frame(), "volume", freq="1h", horizon="1D")
    t = idx[24 * 30]
    assert df.loc[t, "lag_1d"] == y[t - pd.Timedelta("1D")]
    assert df.loc[t, "lag_7d"] == y[t - pd.Timedelta("7D")]
    # 직전 24시간 평균도 t - 1일에서 끝나야 한다(예측 시점에 알 수 있는 값만)
    window = y[t - pd.Timedelta("2D") + pd.Timedelta("1h"): t - pd.Timedelta("1D")]
    assert df.loc[t, "roll24h_mean"] == window.mean()
    for col in lag_columns(df):
        assert df.loc[t, col] <= y[t - pd.Timedelta("1D")]  # 증가 수열 → 미래 값이 섞이면 커진다


def test_lags_respect_gaps_in_time():
    idx = pd.date_range("2026-01-01", periods=24 * 20, freq="h").delete(slice(24 * 5, 24 * 6))
    y = pd.Series(1.0, index=idx, name="volume")
    df = add_lag_features(y.to_frame(), "volume", freq="1h", horizon="1D")
    t = pd.Timestamp("2026-01-07 03:00")  # 1일 전이 빠진 날
    assert np.isnan(df.loc[t, "lag_1d"])


def test_forecast_metrics():
    y = np.array([100.0, 200.0, 0.0])
    p = np.array([110.0, 180.0, 10.0])
    m = summarize(y, p, mape_floor=1.0)
    assert m["MAE"] == (10 + 20 + 10) / 3
    assert abs(m["RMSE"] - np.sqrt((100 + 400 + 100) / 3)) < 1e-9
    assert abs(m["MAPE"] - (10 + 10) / 2) < 1e-9  # 0은 제외
    assert abs(m["WAPE"] - 40 / 300 * 100) < 1e-9
