from src.events.event_detector import EventDetector


def test_event_detector_starts_empty_and_does_not_invent_events():
    detector = EventDetector()
    assert detector.update(1.0, []) == []
    assert detector.events == []
