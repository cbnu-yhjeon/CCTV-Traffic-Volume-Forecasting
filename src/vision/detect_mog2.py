"""배경 차분(MOG2) + 모폴로지 기반 차량 검출.

고정 카메라에서는 움직이는 픽셀이 곧 차량이므로, 전경 마스크의 연결 요소를
차량 후보로 본다. 그림자(MOG2가 127로 표시)는 이진화 단계에서 제거한다.
차종은 박스 면적으로만 구분한다(small/large).
"""

from __future__ import annotations

import cv2
import numpy as np

from src.vision.types import BBox, Detection, in_roi, roi_mask, suppress_contained


def merge_fragments(boxes: list[BBox], max_gap: float, min_x_overlap: float = 0.5) -> list[BBox]:
    """위아래로 붙은 조각 상자를 합친다.

    차체 색이 노면과 비슷하면 한 차가 앞유리 · 차체처럼 두 덩어리로 쪼개진다.
    가로로 많이 겹치고(작은 상자 폭의 min_x_overlap 이상) 세로 간격이 max_gap 이하면 같은 차로 본다.
    """
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                x_overlap = min(a[2], b[2]) - max(a[0], b[0])
                min_w = min(a[2] - a[0], b[2] - b[0])
                gap = max(a[1], b[1]) - min(a[3], b[3])  # 음수면 세로로 겹침
                if min_w > 0 and x_overlap / min_w >= min_x_overlap and gap <= max_gap:
                    boxes[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


class MOG2Detector:
    def __init__(
        self,
        roi: list[list[int]] | None = None,
        scale: float = 0.5,
        history: int = 500,
        var_threshold: float = 32.0,
        detect_shadows: bool = True,
        min_area: int = 3000,
        large_area: int = 90000,
        open_kernel: int = 3,
        close_kernel: int = 15,
        warmup_frames: int = 30,
        merge_gap: float | None = None,
    ) -> None:
        self.roi = roi
        self.scale = scale
        self.merge_gap = merge_gap  # None이면 조각 병합을 하지 않는다(원본 해상도 픽셀)
        self.min_area = min_area  # 원본 해상도 기준 면적
        self.large_area = large_area
        self.warmup_frames = warmup_frames
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=detect_shadows
        )
        self.open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        self.close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        self._mask: np.ndarray | None = None
        self._frames = 0
        self.last_foreground: np.ndarray | None = None

    def foreground(self, frame: np.ndarray) -> np.ndarray:
        """축소 해상도의 정제된 전경 마스크(0/255)."""
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 0)
        fg = self.subtractor.apply(small)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)  # 그림자(127) 제거
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.open_k)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.close_k)
        fg = cv2.dilate(fg, self.open_k, iterations=1)
        return fg

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._mask is None:
            self._mask = roi_mask(frame.shape[:2], self.roi)
        fg = self.foreground(frame)
        self.last_foreground = fg
        self._frames += 1
        if self._frames <= self.warmup_frames:  # 배경 모델이 자리 잡기 전에는 버린다
            return []

        _n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        inv = 1.0 / self.scale
        boxes = [(x * inv, y * inv, (x + w) * inv, (y + h) * inv) for x, y, w, h, _ in stats[1:]]
        if self.merge_gap is not None:
            # 잡음 점은 버리고, 작은 조각은 병합 후보로 남긴다
            boxes = [b for b, s in zip(boxes, stats[1:]) if s[4] * inv * inv >= self.min_area / 4]
            boxes = merge_fragments(boxes, self.merge_gap)
        detections: list[Detection] = []
        for x1, y1, x2, y2 in boxes:
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < self.min_area:
                continue
            det = Detection(
                bbox=(x1, y1, x2, y2),
                cls="large" if box_area >= self.large_area else "small",
            )
            if in_roi(det.center, self._mask):
                detections.append(det)
        return suppress_contained(detections) if self.merge_gap is not None else detections
