import cv2
import numpy as np

from src.events.goal_motion import GoalMotion
from src.events.goal_mouth import GoalMouth


def run(y, cut=False):
    goal = GoalMotion(GoalMouth((640, 300), (640, 360), (500, 330)))
    rng = np.random.default_rng(42)
    background = rng.integers(30, 95, (720, 1280, 3), dtype=np.uint8)
    events = []
    for i in range(25):
        frame = background.copy()
        if i >= 6:
            cv2.circle(frame, (530+(i-6)*10, y), 3, (240, 240, 240), -1)
        if cut and i >= 14:
            frame[:] = 0
        events.extend(goal.observe(frame, i/25))
    return events


def test_motion_review_and_interpolated_crossing():
    events = run(330)
    assert [e["event"] for e in events] == ["shot_candidate", "goal_line_crossing_candidate"]
    assert abs(events[-1]["seconds"] - 17/25) < .04


def test_wide_shot_and_cut_do_not_cross():
    assert not run(395)
    assert not any(e["event"] == "goal_line_crossing_candidate" for e in run(330, cut=True))
