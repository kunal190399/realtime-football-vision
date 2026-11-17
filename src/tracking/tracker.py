"""Persistent object tracking using Ultralytics' ByteTrack integration."""
from __future__ import annotations
from typing import Any
from ultralytics import YOLO


class PlayerTracker:
    def __init__(self, weights: str = "yolo11n.pt", confidence: float = 0.25,
                 tracker: str = "bytetrack.yaml", device: str | None = None) -> None:
        self.model = YOLO(weights)
        self.confidence = confidence
        self.tracker = tracker
        self.device = device

    def track(self, frame: Any) -> Any:
        result = self.model.track(frame, persist=True, tracker=self.tracker,
                                  conf=self.confidence, device=self.device,
                                  verbose=False)[0]
        return result.plot()
