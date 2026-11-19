import pytest
from src.events.goal_mouth import GoalMouth, GoalMonitor


def observe(xs, y=360, sources=None, ids=None):
    monitor = GoalMonitor(GoalMouth((1080, 280), (1080, 440), (640, 360)))
    events = []
    for i, x in enumerate(xs):
        events.extend(monitor.observe(i/25, (x, y), sources[i] if sources else "tracked", ids[i] if ids else 1))
    return events


def test_one_approach_and_interpolated_crossing():
    events = observe([1045, 1055, 1065, 1075, 1085])
    assert [e["event"] for e in events] == ["shot_candidate", "goal_line_crossing_candidate"]
    assert events[-1]["seconds"] == pytest.approx(.14)


def test_reverse_wide_and_discontinuous_paths():
    assert not observe([1090, 1080, 1070])
    assert not observe([1060, 1070, 1080], y=500)
    assert not observe([1000, 1110])
    assert not observe([1070, 1078, 1085], sources=["tracked", "motion", "detected"])
    assert not observe([1070, 1085], ids=[1, 2])


def test_invalid_goal():
    with pytest.raises(ValueError):
        GoalMouth((1, 1), (1, 1), (2, 4))
    with pytest.raises(ValueError):
        GoalMouth((1, 1), (2, 2), (3, 3))
