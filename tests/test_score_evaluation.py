import pytest

from src.evaluation.score_updates import Change, evaluate, load_changes


def change(t, team="FRA", score="1-0"):
    return Change(t, team, score)


def test_duplicates_use_closest_match_without_double_counting():
    r = evaluate([change(3)], [change(2.8), change(3.1), change(3, "ARG")], 0.25)
    assert (r["matched"], r["extra"], r["missed"]) == (1, 2, 0)
    assert r["matches"][0]["predicted_seconds"] == 3.1


def test_maximises_matches_then_uses_inclusive_tolerance():
    r = evaluate([change(2), change(2.3)], [change(1.75), change(2.09)], 0.25)
    assert r["matched"] == 2
    assert r["mean_absolute_error_seconds"] == pytest.approx(.23)


def test_empty_and_wrong_score():
    assert evaluate([], [], .25)["f1"] is None
    r = evaluate([change(1)], [change(1, score="2-0")], .25)
    assert r["f1"] == 0 and r["missed"] == r["extra"] == 1
    with pytest.raises(ValueError):
        evaluate([], [], float("nan"))


def test_dashboard_csv_format_and_bad_time(tmp_path):
    path = tmp_path / "review.csv"
    path.write_text("seconds,team,new_score,review_reason\n0.7,fra,2–1,gap\n")
    assert load_changes(path) == [change(.7, score="2-1")]
    path.write_text("seconds,team,new_score\nnan,FRA,1-0\n")
    with pytest.raises(ValueError, match="line 2"):
        load_changes(path)
