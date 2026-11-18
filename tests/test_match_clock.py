import pytest
from src.events.match_clock import review_changes, clock_seconds


@pytest.mark.parametrize(("later", "reason"), [
    ("80:54", "match-clock discontinuity"),
    ("79:20", "match-clock discontinuity"),
    (None, "unreadable clock"),
    ("79:27", ""),
])
def test_match_clock_controls_export(later, reason):
    samples = [(n, (2, 0)) for n in [0, 12, 24]]
    samples.extend((n, (2, 1)) for n in [120, 132, 144])
    accepted, review = review_changes(samples, 50, ("ARG", "FRA"), [{"frame": 120}],
                                      {24: "79:25", 120: later}.get)
    assert (len(accepted), len(review)) == ((0, 1) if reason else (1, 0))
    assert (accepted + review)[0]["review_reason"] == reason


def test_long_absence_kept_for_review_with_plausible_clock():
    samples = [(n, (1, 1)) for n in [0, 12, 24]]
    samples += [(36, None), (48, (0, 4)), (60, None)]
    samples += [(n, (2, 1)) for n in [180, 192, 204]]
    accepted, review = review_changes(samples, 50, ("ARG", "FRA"), [],
                                      {24: "116:00", 180: "116:03"}.get)
    assert not accepted
    assert [r["review_reason"] for r in review] == ["scoreboard gap"]
    assert clock_seconds("79:61") is None
