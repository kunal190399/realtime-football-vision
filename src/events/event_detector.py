"""Starter event heuristics. These are placeholders, not validated match events."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Event:
    timestamp_seconds: float
    label: str
    confidence: float
    explanation: str


class EventDetector:
    """Collects candidate events for later calibrated event classification.

    Generic pretrained COCO weights do not reliably detect a small football ball
    or distinguish shots/goals. This baseline intentionally emits no goal/shot
    claims until a ball detector and pitch/goal calibration are configured.
    """
    def __init__(self) -> None:
        self.events: list[Event] = []

    def update(self, timestamp_seconds: float, detections: list[dict]) -> list[Event]:
        # Future: ball trajectory, goal-mouth crossing, scoreboard OCR and
        # temporal confirmation will be combined here.
        return []
