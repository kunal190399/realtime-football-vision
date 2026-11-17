"""YOLO-based object detection wrapper for football footage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ultralytics import YOLO


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    track_id: int | None = None


class FootballDetector:
    """Runs a pretrained Ultralytics detector and filters useful classes.

    COCO pretrained models detect people but do not have a dedicated football
    class. For reliable ball detection, fine-tune on a sports-specific dataset.
    """

    def __init__(self, weights: str = "yolo11n.pt", confidence: float = 0.25,
                 device: str | None = None) -> None:
        self.model = YOLO(weights)
        self.confidence = confidence
        self.device = device

    def detect(self, frame: Any) -> list[Detection]:
        result = self.model.predict(frame, conf=self.confidence, device=self.device,
                                    verbose=False)[0]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            class_id = int(box.cls.item())
            name = self.model.names[class_id]
            # COCO person class. Sports-specific weights can add ball classes.
            if name not in {"person", "sports ball", "ball"}:
                continue
            coords = tuple(float(v) for v in box.xyxy[0].tolist())
            detections.append(Detection(coords, float(box.conf.item()), class_id, name))
        return detections
