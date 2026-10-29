"""CCTV 스트림(또는 녹화 영상)을 계속 읽어 15분 교통량을 CSV에 쌓는 수집기.

- 원본 영상은 저장하지 않고 집계 숫자만 기록한다(개인정보).
- 구간마다 실제로 본 시간 비율(coverage)을 함께 기록한다. 끊김 · 시점 변화로 덜 본 구간은
  예측 데이터셋을 만들 때 결측으로 처리한다.
- 시점 변화(PTZ 회전 등): 축소 흑백 프레임과 기준 프레임의 상관계수가 임계값 아래로 떨어지면
  배경 모델 · 추적기를 초기화하고 그 구간을 view_change로 표시한다.
- ITS 스트림 URL에는 만료 토큰이 붙으므로 refresh_minutes마다, 그리고 끊길 때마다 다시 받는다.

    # 녹화 영상으로 시험(20초 구간)
    python -m src.collect.run_collector --config configs/sites/traffic_mp4.yaml --freq 20s
    # ITS 실시간(ITS_API_KEY 필요)
    python -m src.collect.run_collector --config configs/sites/its_example.yaml --duration-min 1440
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.vision.pipeline import CountingPipeline, load_config

log = logging.getLogger("collector")


class ViewMonitor:
    """기준 프레임과의 유사도로 카메라 시점 변화를 감지한다."""

    def __init__(self, threshold: float = 0.6, size: tuple[int, int] = (64, 36), alpha: float = 0.02):
        self.threshold = threshold
        self.size = size
        self.alpha = alpha  # 조명 변화를 따라가도록 기준을 천천히 갱신
        self.reference: np.ndarray | None = None

    def _small(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA).astype(np.float32)

    def similarity(self, frame: np.ndarray) -> float:
        small = self._small(frame)
        if self.reference is None:
            self.reference = small
            return 1.0
        a, b = small - small.mean(), self.reference - self.reference.mean()
        denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / denom) if denom > 0 else 1.0

    def check(self, frame: np.ndarray) -> bool:
        """True면 시점이 바뀌었다고 판단(기준 프레임을 새로 잡는다)."""
        sim = self.similarity(frame)
        small = self._small(frame)
        if sim < self.threshold:
            self.reference = small
            return True
        self.reference = (1 - self.alpha) * self.reference + self.alpha * small
        return False


class BinAccumulator:
    """카운트 이벤트와 관측 시간을 고정 간격 구간으로 모아 CSV에 덧붙인다."""

    def __init__(self, path: Path, freq: str = "15min"):
        self.path = Path(path)
        self.freq = pd.tseries.frequencies.to_offset(freq)
        self.counts: dict[pd.Timestamp, Counter] = defaultdict(Counter)
        self.observed: dict[pd.Timestamp, float] = defaultdict(float)
        self.view_change: set[pd.Timestamp] = set()

    def _bin(self, t: pd.Timestamp) -> pd.Timestamp:
        return t.floor(self.freq)

    def observe(self, t: pd.Timestamp, seconds: float) -> None:
        self.observed[self._bin(t)] += seconds

    def add_event(self, t: pd.Timestamp, direction: str, cls: str) -> None:
        c = self.counts[self._bin(t)]
        c["volume"] += 1
        c[direction] += 1
        c[f"cls_{cls}"] += 1

    def mark_view_change(self, t: pd.Timestamp) -> None:
        self.view_change.add(self._bin(t))

    def flush(self, before: pd.Timestamp | None = None) -> pd.DataFrame:
        """before보다 앞선(닫힌) 구간을 CSV에 쓰고 메모리에서 지운다. None이면 전부."""
        keys = sorted(set(self.counts) | set(self.observed))
        if before is not None:
            keys = [k for k in keys if k + self.freq <= before]
        if not keys:
            return pd.DataFrame()
        span = pd.Timedelta(self.freq).total_seconds()
        rows = []
        for k in keys:
            row = {"timestamp": k, **self.counts.pop(k, Counter())}
            row["volume"] = row.get("volume", 0)
            row["coverage"] = round(min(1.0, self.observed.pop(k, 0.0) / span), 3)
            row["view_change"] = int(k in self.view_change)
            self.view_change.discard(k)
            rows.append(row)
        df = pd.DataFrame(rows)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            old = pd.read_csv(self.path, parse_dates=["timestamp"])
            df = pd.concat([old[~old["timestamp"].isin(df["timestamp"])], df])
            df = df.sort_values("timestamp").reset_index(drop=True)
        lead = ["timestamp", "volume", "coverage", "view_change"]
        df = df[lead + sorted(c for c in df.columns if c not in lead)].fillna(0)
        counts = [c for c in df.columns if c not in ("timestamp", "coverage")]
        df[counts] = df[counts].astype(int)
        df.to_csv(self.path, index=False)
        return df


def resolve_source(cfg: dict) -> str:
    """ITS 설정이 있으면 API로 최신 스트림 URL을 받고, 없으면 source(파일 · URL)를 쓴다."""
    its = cfg.get("its")
    if not its:
        return cfg["source"]
    from src.collect.its_cctv import find_by_name, list_cctvs, nearest

    cctvs = list_cctvs(*its["bbox"], road_type=its.get("road_type", "its"))
    if not cctvs:
        raise RuntimeError("ITS API returned no CCTV")
    cctv = find_by_name(cctvs, its["name"]) if its.get("name") else None
    cctv = cctv or nearest(cctvs, *its["near"])
    log.info("ITS CCTV: %s (%.5f, %.5f)", cctv.name, cctv.lat, cctv.lon)
    return cctv.url


def collect(cfg: dict, method: str, freq: str, out: Path, duration_min: float | None,
            stride: int = 1) -> None:
    is_live = bool(cfg.get("its")) or str(cfg.get("source", "")).startswith(("http", "rtsp"))
    refresh = pd.Timedelta(minutes=cfg.get("its", {}).get("refresh_minutes", 30))
    acc = BinAccumulator(out, freq)
    deadline = time.time() + duration_min * 60 if duration_min else None
    file_start = pd.Timestamp(cfg.get("start_time") or pd.Timestamp.now().floor("s"))
    backoff = 5

    while deadline is None or time.time() < deadline:
        source = resolve_source(cfg)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            log.warning("cannot open stream, retry in %ss", backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
            continue
        backoff = 5
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        pipe = CountingPipeline(cfg, method)
        monitor = ViewMonitor(cfg.get("view_change_threshold", 0.6))
        opened_at = pd.Timestamp.now()
        idx = 0
        log.info("opened %s (fps=%.1f)", source if not is_live else "live stream", fps)

        while deadline is None or time.time() < deadline:
            ok, frame = cap.read()
            if not ok:
                break
            now = pd.Timestamp.now() if is_live else file_start + pd.Timedelta(seconds=idx / fps)
            idx += 1
            if idx % stride:
                continue
            acc.observe(now, stride / fps)
            if idx % int(fps) == 0 and monitor.check(frame):
                log.warning("view change at %s: reset background/tracker", now)
                acc.mark_view_change(now)
                pipe = CountingPipeline(cfg, method)
            for e in pipe.step(frame, idx, fps / stride)[2]:
                acc.add_event(now, e.direction, e.cls)
            if idx % int(fps * 60) == 0:
                acc.flush(before=now.floor(acc.freq))
            if is_live and pd.Timestamp.now() - opened_at > refresh:
                log.info("refreshing stream URL")
                break
        cap.release()
        if not is_live:  # 녹화 영상은 한 번 읽으면 끝
            break
    df = acc.flush()
    log.info("saved %s (%d rows)", out, len(pd.read_csv(out)) if out.exists() else 0)
    if len(df):
        print(df.tail(10).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", default="C3", choices=["C1", "C2", "C3"])
    ap.add_argument("--freq", default="15min")
    ap.add_argument("--out", help="기본값: data/counts/<site>_<freq>.csv")
    ap.add_argument("--duration-min", type=float)
    ap.add_argument("--stride", type=int, default=1, help="n 프레임마다 처리(저사양 · 저프레임 스트림)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    out = Path(args.out or f"data/counts/{cfg['name']}_{args.freq}.csv")
    collect(cfg, args.method, args.freq, out, args.duration_min, args.stride)


if __name__ == "__main__":
    main()
