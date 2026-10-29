"""YOLO(ultralytics) 기반 차량 검출. 배경 차분이 흔들리는 야간 · 우천 비교용."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.vision.types import Detection, in_roi, roi_mask, suppress_contained

# COCO 클래스 중 차량만 사용
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
WEIGHTS_DIR = Path("data/external/weights")


class YOLODetector:
    def __init__(
        self,
        weights: str = "yolo11n.pt",
        roi: list[list[int]] | None = None,
        conf: float = 0.3,
        iou: float = 0.5,
        imgsz: int = 960,
        device: str | None = None,
        agnostic_nms: bool = False,
        suppress_contained: bool = False,
    ) -> None:
        from ultralytics import YOLO

        path = Path(weights)
        if not path.exists() and path.parent == Path("."):
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            path = WEIGHTS_DIR / weights  # 없으면 ultralytics가 이 경로로 내려받는다
        self.model = YOLO(str(path))
        self.roi = roi
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        # 클래스 무관 NMS: 같은 차에 car · truck 상자가 함께 남는 중복을 막는다
        self.agnostic_nms = agnostic_nms
        self.suppress_contained = suppress_contained
        self._mask: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._mask is None:
            self._mask = roi_mask(frame.shape[:2], self.roi)
        result = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=list(VEHICLE_CLASSES),
            device=self.device,
            agnostic_nms=self.agnostic_nms,
            verbose=False,
        )[0]
        boxes = result.boxes
        detections: list[Detection] = []
        for xyxy, score, cls_id in zip(
            boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()
        ):
            det = Detection(
                bbox=tuple(float(v) for v in xyxy),
                score=float(score),
                cls=VEHICLE_CLASSES[int(cls_id)],
            )
            if in_roi(det.center, self._mask):
                detections.append(det)
        return suppress_contained(detections) if self.suppress_contained else detections
