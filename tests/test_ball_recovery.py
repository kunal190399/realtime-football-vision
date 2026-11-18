from types import SimpleNamespace

import numpy as np

from src.tracking.ball_motion import BallMotion
from src.tracking.observations import Track
from src.tracking.tracker import PlayerTracker


def pitch(x):
    frame = np.full((80, 120, 3), (25, 105, 40), np.uint8)
    frame[36:45, x-4:x+5] = 235
    return frame


def test_motion_follows_appearance_for_at_most_five_frames():
    motion = BallMotion(50)
    assert motion.update(pitch(40), 40, 40) is None
    for x in range(42, 51, 2):
        assert motion.update(pitch(x), None, None) == (x, 40)
    assert motion.update(pitch(52), None, None) is None


def test_cut_discards_template_even_if_object_remains():
    motion = BallMotion(50)
    motion.update(pitch(40), 40, 40)
    cut = pitch(42)
    cut[:30] = 240
    cut[50:] = 240
    assert motion.update(cut, None, None) is None
    assert motion.update(pitch(44), None, None) is None


def test_ambiguous_matches_and_empty_frames_stay_unknown():
    motion = BallMotion(50)
    assert motion.update(pitch(40), None, None) is None
    motion.update(pitch(40), 40, 40)
    ambiguous = pitch(50)
    ambiguous[36:45, 26:35] = 235
    assert motion.update(ambiguous, None, None) is None


def test_untracked_detection_survives_empty_tracker_output():
    tracker = PlayerTracker.__new__(PlayerTracker)
    tracker.raw_ball = Track("ball", None, (10, 12, 18, 20), 0.8)
    assert tracker.observations(SimpleNamespace(boxes=None)) == [tracker.raw_ball]
