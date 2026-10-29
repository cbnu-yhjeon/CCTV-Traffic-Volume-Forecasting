"""교통량 + 달력 + 기상 + 지연값 → 학습용 표.

교통량 소스:
- counts: 수집기가 쌓은 CSV(data/counts/<site>_15min.csv). coverage가 낮거나 시점 변화가
  있었던 구간은 결측으로 바꾼다.
- uci_metro: 파이프라인 검증용 대리 데이터(시간별)

기상은 실제 관측(재분석) 값을 쓴다. 실제 운영에서는 예보값이 들어가므로, 기상 효과는
'예보가 완벽할 때'의 상한으로 읽어야 한다.
"""

from __future__ import annotations

import pandas as pd

from src.collect.weather import HOURLY_VARS, fetch_archive
from src.features.calendar import CALENDAR_COLUMNS, add_calendar_features
from src.features.lags import add_lag_features, lag_columns

WEATHER_DERIVED = ["is_rain", "is_snow"]


def load_counts(path: str, freq: str, min_coverage: float = 0.8) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    vol = df["volume"].astype(float)
    bad = (df.get("coverage", 1.0) < min_coverage) | (df.get("view_change", 0) == 1)
    vol[bad] = float("nan")
    full = pd.date_range(vol.index.min(), vol.index.max(), freq=freq)
    vol = vol.reindex(full)
    vol.index.name = "timestamp"
    return vol.rename("volume")


def load_volume(cfg: dict) -> pd.Series:
    if cfg["dataset"] == "uci_metro":
        from src.collect import uci_metro

        return uci_metro.load_volume()
    if cfg["dataset"] == "counts":
        return load_counts(cfg["path"], cfg["freq"], cfg.get("min_coverage", 0.8))
    raise ValueError(f"unknown dataset: {cfg['dataset']}")


def attach_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """시간별 기상을 구간 시작 시각의 '시' 값으로 붙인다(15분 구간 4개가 같은 시간 값을 공유)."""
    w = weather.reindex(df.index.floor("h"))
    w.index = df.index
    out = df.join(w)
    out["is_rain"] = (out["precipitation"] >= 0.1).astype(int)
    out["is_snow"] = (out["snowfall"] > 0).astype(int)
    return out


def build_dataset(cfg: dict, weather: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    volume = load_volume(cfg)
    df = volume.to_frame()
    hol = cfg.get("holidays", {})
    df = add_calendar_features(df, hol.get("country", "KR"), hol.get("subdiv"))
    df = add_lag_features(df, "volume", cfg["freq"], cfg.get("horizon", "1D"))
    if weather is None:
        loc = cfg["location"]
        weather = fetch_archive(
            loc["lat"], loc["lon"],
            df.index.min().strftime("%Y-%m-%d"), df.index.max().strftime("%Y-%m-%d"),
            loc.get("timezone", "Asia/Seoul"),
        )
    df = attach_weather(df, weather)

    calendar = list(CALENDAR_COLUMNS)
    if pd.Timedelta(pd.tseries.frequencies.to_offset(cfg["freq"])) >= pd.Timedelta("1h"):
        calendar.remove("slot")  # 시간 단위 데이터에서는 hour와 같다
    groups = {
        "calendar": calendar,
        "weather": [c for c in HOURLY_VARS if c in df.columns] + WEATHER_DERIVED,
        "lag": lag_columns(df),
    }
    return df, groups
