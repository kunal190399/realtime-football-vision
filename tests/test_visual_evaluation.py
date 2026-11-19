from src.evaluation.visual_events import evaluate


def test_duplicate_predictions_and_absent_class():
    label = dict(event="shot_candidate", seconds=1)
    report = evaluate([label], [label, dict(event="shot_candidate", seconds=1.4)])
    assert report["shot_candidate"]["precision"] == .5
    assert report["shot_candidate"]["timing_errors_seconds"] == [0.]
    assert report["goal_line_crossing_candidate"]["f1"] is None
