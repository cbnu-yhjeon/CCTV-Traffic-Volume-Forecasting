"""달력 특징: 시각 · 요일 · 공휴일(대체공휴일 포함) · 연휴 앞뒤."""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

CALENDAR_COLUMNS = [
    "hour", "slot", "dow", "month", "day_of_year", "is_weekend", "is_holiday", "is_offday",
    "prev_offday", "next_offday", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
]


def holiday_dates(years: list[int], country: str = "KR", subdiv: str | None = None) -> set:
    return set(holidays.country_holidays(country, subdiv=subdiv, years=years).keys())


def add_calendar_features(df: pd.DataFrame, country: str = "KR", subdiv: str | None = None) -> pd.DataFrame:
    """df.index(DatetimeIndex, 현지 시각)로 달력 특징 열을 붙인다."""
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    dates = idx.normalize()
    years = sorted(set(idx.year) | {idx.year.min() - 1, idx.year.max() + 1})
    hol = holiday_dates(years, country, subdiv)
    is_hol = lambda d: np.array([x.date() in hol for x in d], dtype=int)
    is_off = lambda d: ((d.dayofweek >= 5).astype(int) | is_hol(d))

    out["hour"] = idx.hour
    out["slot"] = idx.hour * 4 + idx.minute // 15  # 15분 단위 하루 중 순번
    out["dow"] = idx.dayofweek
    out["month"] = idx.month
    out["day_of_year"] = idx.dayofyear
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_holiday"] = is_hol(dates)
    out["is_offday"] = is_off(dates)
    out["prev_offday"] = is_off(dates - pd.Timedelta("1D"))  # 연휴 다음 날(복귀일)
    out["next_offday"] = is_off(dates + pd.Timedelta("1D"))  # 연휴 전날(귀성)
    frac_hour = idx.hour + idx.minute / 60
    out["hour_sin"] = np.sin(2 * np.pi * frac_hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * frac_hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    out["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    return out
