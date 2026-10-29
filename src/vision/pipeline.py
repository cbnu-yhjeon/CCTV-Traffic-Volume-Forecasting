"""검출 → 추적 → 카운팅을 묶은 파이프라인과 실험 구성(C1~C3)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.vision.counter import CountEvent, LineCounter
from src.vision.tracker import IoUTracker, LKTracker, Track
from src.vision.types import Detection

# 실험 ID → (검출기, 추적기)
METHODS = {
    "C1": ("mog2", "iou"),
    "C2": ("mog2", "lk"),
    "C3": ("yolo", "iou"),
}


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_detector(kind: str, cfg: dict):
    if kind == "mog2":
        from src.vision.detect_mog2 import MOG2Detector

        return MOG2Detector(roi=cfg.get("roi"), **cfg.get("mog2", {}))
    if kind == "yolo":
        from src.vision.detect_yolo import YOLODetector

        return YOLODetector(roi=cfg.get("roi"), **cfg.get("yolo", {}))
    raise ValueError(f"unknown detector: {kind}")


def build_tracker(kind: str, cfg: dict) -> IoUTracker:
    params = cfg.get("tracker", {})
    return LKTracker(**params) if kind == "lk" else IoUTracker(**params)


def build_counter(cfg: dict) -> LineCounter:
    line = cfg["line"]
    return LineCounter(
        p1=line["p1"],
        p2=line["p2"],
        directions=cfg.get("directions"),
        lanes=cfg.get("lanes"),
        anchor=line.get("anchor", "center"),
    )


class CountingPipeline:
    def __init__(self, cfg: dict, method: str) -> None:
        det_kind, trk_kind = METHODS[method]
        self.cfg = cfg
        self.method = method
        self.detector = build_detector(det_kind, cfg)
        self.tracker = build_tracker(trk_kind, cfg)
        self.counter = build_counter(cfg)

    def step(
        self, frame: np.ndarray, frame_idx: int, fps: float
    ) -> tuple[list[Detection], list[Track], list[CountEvent]]:
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame)
        events = self.counter.update(tracks, frame_idx, fps)
        return detections, tracks, events


def iter_frames(source: str, stride: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video source: {source}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


COLORS = {"small": (0, 200, 255), "car": (0, 200, 255), "large": (255, 80, 0), "truck": (255, 80, 0),
          "bus": (255, 0, 200), "motorcycle": (0, 255, 0)}


def draw(frame: np.ndarray, cfg: dict, tracks: list[Track], total: int, method: str) -> np.ndarray:
    out = frame.copy()
    if cfg.get("roi"):
        cv2.polylines(out, [np.asarray(cfg["roi"], np.int32)], True, (255, 255, 0), 2)
    p1, p2 = cfg["line"]["p1"], cfg["line"]["p2"]
    cv2.line(out, tuple(p1), tuple(p2), (0, 0, 255), 4)
    for t in tracks:
        if not t.confirmed:
            continue
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        color = COLORS.get(t.cls, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if t.misses == 0 else 1)
        cv2.putText(out, f"#{t.id} {t.cls}", (x1, max(30, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.rectangle(out, (0, 0), (520, 70), (0, 0, 0), -1)
    cv2.putText(out, f"{method}  count={total}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return out
