"""Open-Meteo 시간별 기상 수집(인증키 불필요).

- archive: 과거 재분석 자료(ERA5, 약 5일 지연) → 학습용
- forecast: 최근 + 예보 → 연속 수집 · 실시간 예측용

    python -m src.collect.weather --lat 36.63 --lon 127.49 --start 2026-10-01 --end 2026-10-28
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "cloud_cover",
    "wind_speed_10m",
    "weather_code",
]
CACHE_DIR = Path("data/external/weather")


def _get(url: str, params: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _to_frame(payload: dict) -> pd.DataFrame:
    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    # 서머타임 전환 시 같은 현지 시각이 두 번 나올 수 있어 첫 값만 둔다
    return df.drop_duplicates("time").set_index("time").sort_index()


def fetch_archive(
    lat: float, lon: float, start: str, end: str, timezone: str = "Asia/Seoul",
    variables: list[str] = HOURLY_VARS, cache_dir: Path | None = CACHE_DIR,
) -> pd.DataFrame:
    """[start, end] 기간 시간별 기상. 연 단위로 나눠 받고 캐시한다. index = 현지 시각(naive)."""
    frames = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        s = max(pd.Timestamp(start), pd.Timestamp(f"{year}-01-01")).strftime("%Y-%m-%d")
        e = min(pd.Timestamp(end), pd.Timestamp(f"{year}-12-31")).strftime("%Y-%m-%d")
        cache = None
        if cache_dir is not None:
            cache = Path(cache_dir) / f"archive_{lat:.3f}_{lon:.3f}_{s}_{e}.csv"
            if cache.exists():
                frames.append(pd.read_csv(cache, index_col="time", parse_dates=["time"]))
                continue
        payload = _get(ARCHIVE_URL, {
            "latitude": lat, "longitude": lon, "start_date": s, "end_date": e,
            "hourly": ",".join(variables), "timezone": timezone,
        })
        df = _to_frame(payload)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache)
        frames.append(df)
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated()]


def fetch_forecast(
    lat: float, lon: float, past_days: int = 7, forecast_days: int = 2,
    timezone: str = "Asia/Seoul", variables: list[str] = HOURLY_VARS,
) -> pd.DataFrame:
    payload = _get(FORECAST_URL, {
        "latitude": lat, "longitude": lon, "past_days": past_days, "forecast_days": forecast_days,
        "hourly": ",".join(variables), "timezone": timezone,
    })
    return _to_frame(payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--timezone", default="Asia/Seoul")
    ap.add_argument("--forecast", action="store_true", help="최근 7일 + 예보 2일")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.forecast:
        df = fetch_forecast(args.lat, args.lon, timezone=args.timezone)
    else:
        df = fetch_archive(args.lat, args.lon, args.start, args.end, args.timezone)
    print(df.describe().T[["mean", "min", "max"]])
    if args.out:
        df.to_csv(args.out)


if __name__ == "__main__":
    main()
