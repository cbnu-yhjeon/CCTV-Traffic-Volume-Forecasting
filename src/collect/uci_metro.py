"""대리 데이터: UCI Metro Interstate Traffic Volume (미국 I-94 서행 방향, ATR 301 지점).

CCTV 연속 수집이 몇 주 쌓이기 전까지 예측 파이프라인(P0~P4, SHAP)을 검증하는 데 쓴다.
2012-10 ~ 2018-09 시간별 교통량. 기상 열은 버리고 Open-Meteo로 다시 붙여, CCTV 지점과 같은
기상 · 달력 파이프라인을 그대로 쓴다.

https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
PATH = Path("data/external/Metro_Interstate_Traffic_Volume.csv.gz")
# ATR 301: I-94, 미니애폴리스 ~ 세인트폴 중간
LAT, LON, TIMEZONE = 44.963, -93.196, "America/Chicago"


def download(path: Path = PATH) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        path.write_bytes(zf.read(path.name))
    return path


def load_volume(path: Path = PATH) -> pd.Series:
    """시간별 교통량 Series(index = 현지 시각, 빈 시간은 NaN으로 채운 고정 간격)."""
    df = pd.read_csv(download(path), parse_dates=["date_time"])
    # 기상 상태가 여러 개인 시간은 행이 중복된다. 교통량은 같으므로 하나만 남긴다
    s = df.drop_duplicates("date_time").set_index("date_time")["traffic_volume"].sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(full).astype(float)
    s.index.name = "timestamp"
    s.name = "volume"
    return s
