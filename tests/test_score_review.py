from unittest.mock import patch

from streamlit.testing.v1 import AppTest


def test_review_export_requires_selection_and_resets_on_upload_change():
    row = dict(frame=120, seconds=2.4, team="FRA", new_score="2–1",
               previous_clock="79:25", new_clock="80:54", review_reason="match-clock discontinuity")
    report = dict(events=[], review=[row], samples=10, readable=8, duration_seconds=8)
    app = AppTest.from_string("""
import streamlit as st
from app.score_review import show_score_scan
video = st.file_uploader("Clip")
show_score_scan(video, None, None, True, 8, 2)
""")
    with patch("app.score_review.scan_scoreboard", return_value=report) as scan, \
         patch("app.score_review.export_highlights", return_value=b"archive") as export:
        app.run()
        app.get("file_uploader")[0].upload("one.mp4", b"one", "video/mp4").run()
        app.button[0].click().run()
        assert not app.exception
        assert app.button[-1].disabled
        export.assert_not_called()
        app.multiselect[0].set_value(app.multiselect[0].options).run()
        app.button[-1].click().run()
        assert not app.exception
        assert export.call_args.args[1] == [row]
        assert app.session_state["score_scan"]["manual_zip"] == b"archive"
        scan.assert_called_once()
        app.multiselect[0].set_value([]).run()
        assert "manual_zip" not in app.session_state["score_scan"]
        app.get("file_uploader")[0].upload("two.mp4", b"two", "video/mp4").run()
        assert "score_scan" not in app.session_state
