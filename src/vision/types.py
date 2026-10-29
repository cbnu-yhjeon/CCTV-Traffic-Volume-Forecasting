"""비전 파이프라인에서 공유하는 자료형."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 (원본 해상도 픽셀)


@dataclass
class Detection:
    bbox: BBox
    score: float = 1.0
    cls: str = "vehicle"

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2


def roi_mask(shape: tuple[int, int], polygon: list[list[int]] | None) -> np.ndarray | None:
    """ROI 다각형을 0/255 마스크로 만든다. polygon이 없으면 None."""
    if not polygon:
        return None
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
    return mask


def _area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a: BBox, b: BBox) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def suppress_contained(dets: list[Detection], thr: float = 0.8) -> list[Detection]:
    """다른(더 큰) 상자에 면적의 thr 이상이 들어가 있는 상자를 지운다.

    한 차량에 차체 · 지붕 짐칸처럼 상자가 겹쳐 나오는 중복 검출을 없앤다.
    """
    order = sorted(dets, key=lambda d: _area(d.bbox), reverse=True)
    kept: list[Detection] = []
    for d in order:
        a = _area(d.bbox)
        if a > 0 and all(_intersection(d.bbox, k.bbox) / a < thr for k in kept):
            kept.append(d)
    return kept


def in_roi(point: tuple[float, float], mask: np.ndarray | None) -> bool:
    if mask is None:
        return True
    x, y = int(point[0]), int(point[1])
    h, w = mask.shape
    return 0 <= x < w and 0 <= y < h and mask[y, x] > 0
