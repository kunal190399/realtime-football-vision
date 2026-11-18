"""Frame-level possession candidates based on visible tracks in image space."""
from math import hypot

from src.tracking.observations import Track


class PossessionEstimator:
    def __init__(self, confirmation_frames: int = 3):
        self.confirmation_frames = confirmation_frames
        self.ball_id = None
        self.player_id = None
        self.streak = 0

    def update(self, tracks: list[Track]) -> dict:
        balls = [track for track in tracks if track.kind == "ball"]
        ball = max(balls, key=lambda b: (b.track_id is not None and
                   b.track_id == self.ball_id, b.confidence), default=None)
        current_ball = ball.track_id if ball else None
        x = y = candidate = None
        if ball:
            left, top, right, bottom = ball.xyxy
            x, y = (left + right) / 2, (top + bottom) / 2
            distances = []
            for player in tracks:
                if player.kind != "player" or player.track_id is None:
                    continue
                left, top, right, bottom = player.xyxy
                height = bottom - top
                if height > 0:
                    distance = hypot(x - (left + right) / 2, y - bottom) / height
                    if distance <= 0.5:
                        distances.append((distance, player.track_id))
            if distances and current_ball is not None:
                candidate = min(distances)[1]
        if candidate is None or candidate != self.player_id or current_ball != self.ball_id:
            self.streak = 0
        if candidate is not None:
            self.streak += 1
        self.ball_id, self.player_id = current_ball, candidate
        owner = candidate if self.streak >= self.confirmation_frames else None
        return dict(ball_track_id=current_ball, ball_x=x, ball_y=y, possession_track_id=owner)
