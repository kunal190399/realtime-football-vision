import pytest

from src.evaluation.runtime import RuntimeProfile


def test_wall_speed_and_latency_use_the_correct_denominators(monkeypatch):
    ticks = iter([1.0, 1.1, 1.5, 3.0])
    monkeypatch.setattr("src.evaluation.runtime.perf_counter", lambda: next(ticks))
    profile = RuntimeProfile()
    profile.record("OCR", 1.0)
    profile.record("OCR", 1.2)
    report = profile.report(200, 50.0, 1280, 720, {"mode": "scoreboard"})
    assert report["source_seconds"] == 4 and report["wall_seconds"] == 2
    assert report["processed_fps"] == 100 and report["playback_speed"] == 2
    stage = report["stages"]["OCR"]
    assert stage["calls"] == 2
    assert stage["total_seconds"] == pytest.approx(.4)
    assert stage["mean_ms"] == pytest.approx(200)
    assert stage["p95_ms"] == pytest.approx(290)
