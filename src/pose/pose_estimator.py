"""Human pose estimation using an Ultralytics pose model."""
from __future__ import annotations

from typing import Any
from ultralytics import YOLO


class PoseEstimator:
    def __init__(self, weights: str = "yolo11n-pose.pt", device: str | None = None) -> None:
        self.model = YOLO(weights)
        self.device = device

    def annotate(self, frame: Any, canvas: Any = None) -> Any:
        result = self.model.predict(frame, device=self.device, verbose=False)[0]
        return result.plot(img=canvas)
