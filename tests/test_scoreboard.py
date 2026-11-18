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


def test_custom_layout_rejects_overlapping_outside_and_invalid_team_codes():
    import pytest
    from src.events.scoreboard import ScoreLayout

    left = (10, 20, 18, 22)
    for teams, right in [(("RED", "RED"), (50, 20, 18, 22)),
                         (("RED", "../"), (50, 20, 18, 22)),
                         (("RED", "BLU"), (15, 20, 18, 22)),
                         (("RED", "BLU"), (1270, 20, 18, 22))]:
        with pytest.raises(ValueError):
            ScoreLayout(teams, (left, right))


def test_custom_layout_treats_blank_cells_as_unreadable():
    import numpy as np
    from src.events.scoreboard import ScoreLayout, ScoreReader

    layout = ScoreLayout(("RED", "BLU"), ((10, 20, 18, 22), (50, 20, 18, 22)))
    reader = ScoreReader(layout=layout)
    assert reader.read(np.full((360, 640, 3), 200, dtype=np.uint8)) is None


def test_custom_team_codes_follow_the_incrementing_side():
    monitor = ScoreMonitor(("RED", "BLU"))
    events = [monitor.observe(score, frame, 4)
              for frame, score in enumerate([(0, 0)]*3 + [(0, 1)]*3 + [(1, 1)]*3)]
    assert [e.team for e in events if e] == ["BLU", "RED"]
