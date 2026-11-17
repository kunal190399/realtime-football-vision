"""Person-only tracking using Ultralytics ByteTrack.

The football configuration is an experimental starting point. Track quality
must be measured on representative footage before making accuracy claims.
"""
from __future__ import annotations

from typing import Any

from ultralytics import YOLO


class PlayerTracker:
    def __init__(
        self,
        weights: str = "yolo11n.pt",
        confidence: float = 0.25,
        tracker: str = "configs/bytetrack_football.yaml",
        device: str | None = None,
    ) -> None:
        self.model = YOLO(weights)
        self.confidence = confidence
        self.tracker = tracker
        self.device = device

    def track(self, frame: Any) -> Any:
        # COCO class 0 is person. Restricting classes avoids assigning player
        # track IDs to unrelated objects in the scene.
        result = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.confidence,
            classes=[0],
            device=self.device,
            verbose=False,
        )[0]
        return result.plot()
