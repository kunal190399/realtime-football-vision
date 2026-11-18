import pytest

from src.events.highlights import clip_window


def test_clip_windows_include_the_event_and_stop_at_video_boundaries():
    assert clip_window(1, 12, 8, 2) == (0, 3)
    assert clip_window(11.5, 12, 8, 2) == (3.5, 12)
    assert clip_window(10, 20, 8, 2) == (2, 12)


def test_invalid_event_and_empty_tail_are_rejected():
    with pytest.raises(ValueError):
        clip_window(21, 20, 8, 2)
    with pytest.raises(ValueError):
        clip_window(8, 20, 8, 0)
