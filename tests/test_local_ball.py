from unittest.mock import Mock

import cv2
import numpy as np

from src.detection.local_ball import LocalBallDetector
from src.pipeline.realtime_pipeline import RealtimePipeline
from src.tracking.observations import Track


def test_bundled_model_handles_blank_and_low_contrast_frames():
    detector = LocalBallDetector()
    frame = np.full((360, 640, 3), (45, 125, 60), np.uint8)
    assert detector.detect(frame) is None
    cv2.circle(frame, (320, 200), 2, (205, 220, 205), -1)
    strongest = detector.strongest(frame)
    assert strongest is not None
    x1, y1, x2, y2 = strongest.xyxy
    assert x1 < 320 < x2 and y1 < 200 < y2
    assert detector.detect(frame, threshold=1.) is None


def test_recovery_marks_only_accepted_fallback_observations(tmp_path, monkeypatch):
    frame = np.full((180, 320, 3), (40, 130, 45), np.uint8)
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 25, (320, 180))
    for _ in range(2):
        writer.write(frame)
    writer.release()
    native = Track("ball", 5, (50, 50, 60, 60), .95)
    appearance = Track("ball", None, (100, 100, 110, 110), .94)
    detector = Mock()
    detector.detect.return_value = appearance
    monkeypatch.setattr("src.pipeline.realtime_pipeline.LocalBallDetector", lambda: detector)
    pipe = RealtimePipeline.__new__(RealtimePipeline)
    pipe.settings = {}
    pipe.tracker = Mock()
    pipe.tracker.observations.side_effect = [[native], []]
    pipe.pose = Mock()
    pipe.pose.annotate.side_effect = lambda frame, canvas: frame.copy()
    report = pipe.process_video(str(source), str(tmp_path/"out.mp4"), possession=True, local_ball=True)
    assert [row["ball_source"] for row in report["timeline"]] == ["tracked", "appearance"]
    assert report["timeline"][1]["possession_track_id"] is None
    detector.detect.assert_called_once()
