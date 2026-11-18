"""Person-only tracking using Ultralytics ByteTrack.

The football configuration is an experimental starting point. Track quality
must be measured on representative footage before making accuracy claims.
"""
from __future__ import annotations

from typing import Any

from ultralytics import YOLO

from src.tracking.observations import Track


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
        self.kinds = {}
        for class_id, name in self.model.names.items():
            if name.lower() in {"player", "person"}:
                self.kinds[class_id] = "player"
            elif name.lower() in {"football", "ball", "sports ball"}:
                self.kinds[class_id] = "ball"

    def track(self, frame: Any) -> Any:
        return self.track_result(frame).plot()

    def track_result(self, frame: Any, include_ball: bool = False) -> Any:
        classes = [i for i, kind in self.kinds.items() if include_ball or kind == "player"]
        if "player" not in self.kinds.values():
            raise ValueError("Detection weights require a person or player class.")
        if include_ball and "ball" not in self.kinds.values():
            raise ValueError("Possession analysis requires a ball class in the detection weights.")
        return self.model.track(
            frame, persist=True, tracker=self.tracker, conf=self.confidence,
            classes=classes, device=self.device, verbose=False,
        )[0]

    def observations(self, result: Any) -> list[Track]:
        if result.boxes is None:
            return []
        tracks = []
        for box in result.boxes.cpu():
            identity = int(box.id.item()) if box.id is not None else None
            tracks.append(Track(self.kinds[int(box.cls.item())], identity,
                                tuple(box.xyxy[0].tolist()), float(box.conf.item())))
        return tracks
