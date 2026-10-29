"""IoU 기반 다중 객체 추적기(SORT의 칼만 필터를 등속 예측으로 단순화)와 LK 보완판.

- IoUTracker: 예측 박스와 검출 박스의 IoU로 헝가리안 매칭.
- LKTracker: 매칭에 실패한 트랙을 Lucas-Kanade 광류로 옮겨 가려짐 · 검출 누락 구간을 메운다.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.vision.types import BBox, Detection


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _shift(bbox: BBox, dx: float, dy: float) -> BBox:
    return (bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy)


@dataclass
class Track:
    id: int
    bbox: BBox
    hits: int = 1
    misses: int = 0  # 마지막 검출 매칭 이후 지난 프레임 수
    velocity: tuple[float, float] = (0.0, 0.0)
    cls_votes: Counter = field(default_factory=Counter)
    centers: deque = field(default_factory=lambda: deque(maxlen=30))
    min_hits: int = 3

    @property
    def confirmed(self) -> bool:
        return self.hits >= self.min_hits

    @property
    def cls(self) -> str:
        return self.cls_votes.most_common(1)[0][0] if self.cls_votes else "vehicle"

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def predicted(self) -> BBox:
        return _shift(self.bbox, *self.velocity)

    def update(self, det: Detection) -> None:
        old = self.center
        self.bbox = det.bbox
        new = self.center
        # 속도는 지수 이동 평균으로 부드럽게
        steps = self.misses + 1
        vx, vy = (new[0] - old[0]) / steps, (new[1] - old[1]) / steps
        self.velocity = (0.5 * self.velocity[0] + 0.5 * vx, 0.5 * self.velocity[1] + 0.5 * vy)
        self.hits += 1
        self.misses = 0
        self.cls_votes[det.cls] += 1
        self.centers.append(new)

    def coast(self, bbox: BBox | None = None) -> None:
        """검출 없이 한 프레임 진행. bbox가 주어지면(광류) 그 위치로, 없으면 등속 예측."""
        self.bbox = bbox if bbox is not None else self.predicted()
        self.misses += 1
        self.centers.append(self.center)


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10, min_hits: int = 3) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[Track] = []
        self._next_id = 1

    def _match(self, detections: list[Detection]) -> tuple[list[tuple[int, int]], set, set]:
        if not self.tracks or not detections:
            return [], set(range(len(self.tracks))), set(range(len(detections)))
        cost = np.zeros((len(self.tracks), len(detections)))
        for i, t in enumerate(self.tracks):
            pred = t.predicted()
            for j, d in enumerate(detections):
                cost[i, j] = -iou(pred, d.bbox)
        rows, cols = linear_sum_assignment(cost)
        matches = [(r, c) for r, c in zip(rows, cols) if -cost[r, c] >= self.iou_threshold]
        matched_t = {r for r, _ in matches}
        matched_d = {c for _, c in matches}
        return (
            matches,
            set(range(len(self.tracks))) - matched_t,
            set(range(len(detections))) - matched_d,
        )

    def _coast_unmatched(self, unmatched: set, frame: np.ndarray | None) -> None:
        for i in unmatched:
            self.tracks[i].coast()

    def update(self, detections: list[Detection], frame: np.ndarray | None = None) -> list[Track]:
        matches, unmatched_t, unmatched_d = self._match(detections)
        for ti, di in matches:
            self.tracks[ti].update(detections[di])
        self._coast_unmatched(unmatched_t, frame)
        for di in sorted(unmatched_d):
            det = detections[di]
            track = Track(id=self._next_id, bbox=det.bbox, min_hits=self.min_hits)
            track.cls_votes[det.cls] += 1
            track.centers.append(track.center)
            self.tracks.append(track)
            self._next_id += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
        return [t for t in self.tracks if t.misses == 0 or t.confirmed]


class LKTracker(IoUTracker):
    """매칭되지 않은 확정 트랙을 LK 광류로 이동시킨다."""

    def __init__(self, *args, lk_scale: float = 0.5, min_points: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lk_scale = lk_scale
        self.min_points = min_points
        self._prev_gray: np.ndarray | None = None
        self._gray: np.ndarray | None = None
        self.lk_params = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        }

    def update(self, detections: list[Detection], frame: np.ndarray | None = None) -> list[Track]:
        if frame is not None:
            small = cv2.resize(frame, None, fx=self.lk_scale, fy=self.lk_scale)
            self._prev_gray, self._gray = self._gray, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return super().update(detections, frame)

    def _flow_bbox(self, track: Track) -> BBox | None:
        if self._prev_gray is None or self._gray is None:
            return None
        s = self.lk_scale
        h, w = self._prev_gray.shape
        x1, y1, x2, y2 = (int(v * s) for v in track.bbox)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        mask = np.zeros_like(self._prev_gray)
        mask[y1:y2, x1:x2] = 255
        pts = cv2.goodFeaturesToTrack(
            self._prev_gray, maxCorners=50, qualityLevel=0.01, minDistance=3, mask=mask
        )
        if pts is None or len(pts) < self.min_points:
            return None
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, self._gray, pts, None, **self.lk_params)
        good = status.reshape(-1) == 1
        if good.sum() < self.min_points:
            return None
        d = np.median((nxt[good] - pts[good]).reshape(-1, 2), axis=0) / s
        return _shift(track.bbox, float(d[0]), float(d[1]))

    def _coast_unmatched(self, unmatched: set, frame: np.ndarray | None) -> None:
        for i in unmatched:
            track = self.tracks[i]
            track.coast(self._flow_bbox(track) if track.confirmed else None)
