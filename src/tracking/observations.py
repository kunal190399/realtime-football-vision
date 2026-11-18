from dataclasses import dataclass


@dataclass
class Track:
    kind: str
    track_id: int | None
    xyxy: tuple[float, float, float, float]
    confidence: float
