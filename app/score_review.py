"""Keep scoreboard results and manual highlight selections in the current session."""
import csv
import io
import tempfile
from pathlib import Path
from uuid import uuid4
from zipfile import ZipFile

import streamlit as st

from src.events.highlights import export_highlights
from src.events.scoreboard import scan_scoreboard


def export_upload(uploaded, events, duration, lead, tail):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / ("input" + Path(uploaded.name).suffix)
        path.write_bytes(uploaded.getbuffer())
        return export_highlights(path, events, duration, lead, tail)


def download_rows(rows, label, filename):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    st.download_button(label, buffer.getvalue(), file_name=filename, mime="text/csv", on_click="ignore")


def manual_review(uploaded, rows, duration, lead, tail, state):
    by_label = {f"{r['seconds']:.2f} s · {r['team']} · {r['new_score']}": r for r in rows}
    key = state["id"]
    label = st.selectbox("Change to replay", list(by_label), key=f"preview-{key}")
    event = by_label[label]
    st.caption(f"Review reason: {event['review_reason']}. "
               f"Match clock: {event['previous_clock'] or 'unreadable'} → {event['new_clock'] or 'unreadable'}.")
    if st.button("Preview flagged clip"):
        with st.spinner("Preparing preview..."):
            archive = export_upload(uploaded, [event], duration, lead, tail)
            with ZipFile(io.BytesIO(archive)) as bundle:
                clip = next(name for name in bundle.namelist() if name.endswith(".mp4"))
                state["preview"] = (label, bundle.read(clip))
    if state.get("preview", (None,))[0] == label:
        st.video(state["preview"][1])

    selected = st.multiselect("Include these flagged clips in a separate export",
                              list(by_label), key=f"selection-{key}")
    labels = tuple(label for label in by_label if label in selected)
    if state.get("selection") != labels:
        state.pop("manual_zip", None)
        state["selection"] = labels
    if st.button("Build selected highlights", disabled=not labels):
        with st.spinner("Exporting selected highlights..."):
            state["manual_zip"] = export_upload(uploaded, [by_label[label] for label in labels],
                                                duration, lead, tail)
    if state.get("manual_zip"):
        st.download_button("Download selected highlights", state["manual_zip"],
                           file_name="reviewed-highlights.zip", mime="application/zip", on_click="ignore")


def show_score_scan(uploaded, layout, clock_box, make_clips, lead, tail):
    context = (uploaded.file_id if uploaded else None, layout, clock_box, make_clips, lead, tail)
    state = st.session_state.get("score_scan")
    if state and state["context"] != context:
        del st.session_state["score_scan"]
        state = None
    if uploaded and st.button("Scan scoreboard", type="primary"):
        st.session_state.pop("score_scan", None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ("input" + Path(uploaded.name).suffix)
            path.write_bytes(uploaded.getbuffer())
            with st.spinner("Reading the scoreboard..."):
                report = scan_scoreboard(str(path), layout, clock_box)
            state = {"id": uuid4().hex, "context": context, "report": report}
            st.session_state["score_scan"] = state
            if make_clips and report["events"]:
                try:
                    with st.spinner("Exporting highlight clips..."):
                        state["auto_zip"] = export_highlights(path, report["events"],
                                                              report["duration_seconds"], lead, tail)
                except (OSError, RuntimeError, ValueError) as exc:
                    st.warning(f"Highlight export failed: {exc}")
    if not state:
        return
    report = state["report"]
    if state.get("auto_zip"):
        st.download_button("Download highlights and index", state["auto_zip"], file_name="highlights.zip",
                           mime="application/zip", on_click="ignore")
    st.write(f"{len(report['events'])} score updates. "
             f"Score readable in {report['readable']} of {report['samples']} samples.")
    if report["events"]:
        st.dataframe(report["events"], hide_index=True)
        download_rows(report["events"], "Download score updates", "score-updates.csv")
    elif not report["readable"]:
        st.info("No readable scores. Check the scoreboard layout.")
    if report["review"]:
        st.subheader("Score changes requiring review")
        st.caption(f"Automatic highlights exclude these changes. Replay window: {lead} seconds before "
                   f"the new score and {tail} seconds after.")
        st.dataframe(report["review"], hide_index=True)
        download_rows(report["review"], "Download clock review", "clock-review.csv")
        manual_review(uploaded, report["review"], report["duration_seconds"], lead, tail, state)
