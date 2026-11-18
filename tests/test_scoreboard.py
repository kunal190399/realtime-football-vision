from src.events.scoreboard import ScoreMonitor


def scan(scores):
    monitor = ScoreMonitor()
    return [event for frame, score in enumerate(scores)
            if (event := monitor.observe(score, frame, 4)) is not None]


def test_requires_three_consecutive_reads_and_timestamps_first():
    events = scan([(0, 0)] * 3 + [(0, 1), None] + [(0, 1)] * 5)
    assert len(events) == 1
    assert (events[0].team, events[0].frame, events[0].confirmed_seconds) == ("AUS", 5, 1.75)


def test_jumps_and_replays_do_not_create_goals():
    assert scan([(0, 0)] * 3 + [(2, 0)] * 3 + [(1, 0)] * 3 + [(2, 0)] * 3) == []


def test_rebaseline_after_scoreboard_disappears():
    assert scan([(1, 1)] * 3 + [None] * 10 + [(2, 1)] * 3) == []
