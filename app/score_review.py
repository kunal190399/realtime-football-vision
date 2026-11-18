"""Keep scoreboard results and manual highlight selections in the current session."""
import csv
import io
import tempfile
from pathlib import Path
from uuid import uuid4
from zipfile import ZipFile

import streamlit as st

from src.events.highlights import clip_window, export_highlights
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
    trims = state.setdefault("trims", {})
    before, after = trims.get(label, (lead, tail))
    left, right = st.columns(2)
    before = left.number_input("Lead-in for this clip (seconds)", min_value=0.0,
                               value=float(before), step=0.5, key=f"lead-{key}-{label}")
    after = right.number_input("Follow-up for this clip (seconds)", min_value=0.1,
                               value=float(after), step=0.5, key=f"tail-{key}-{label}")
    trims[label] = before, after
    start, end = clip_window(event["seconds"], duration, before, after)
    st.caption(f"Source interval {start:.2f}–{end:.2f} s · duration {end-start:.2f} s")
    preview_key = (label, before, after)
    def trimmed(name):
        window = trims.get(name, (lead, tail))
        if window == (lead, tail):
            return by_label[name]
        return dict(by_label[name], lead_seconds=window[0], tail_seconds=window[1])

    if st.button("Preview flagged clip"):
        with st.spinner("Preparing preview..."):
            archive = export_upload(uploaded, [trimmed(label)], duration, lead, tail)
            with ZipFile(io.BytesIO(archive)) as bundle:
                clip = next(name for name in bundle.namelist() if name.endswith(".mp4"))
                state["preview"] = (preview_key, bundle.read(clip))
    if state.get("preview", (None,))[0] == preview_key:
        st.video(state["preview"][1])

    selected = st.multiselect("Include these flagged clips in a separate export",
                              list(by_label), key=f"selection-{key}")
    labels = tuple(label for label in by_label if label in selected)
    selection = tuple((name, trims.get(name, (lead, tail))) for name in labels)
    if state.get("selection") != selection:
        state.pop("manual_zip", None)
        state["selection"] = selection
    if st.button("Build selected highlights", disabled=not labels):
        with st.spinner("Exporting selected highlights..."):
            state["manual_zip"] = export_upload(uploaded, [trimmed(name) for name in labels],
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
        st.caption(f"Automatic highlights exclude these changes. Default window: {lead} seconds before "
                   f"the new score and {tail} seconds after.")
        st.dataframe(report["review"], hide_index=True)
        download_rows(report["review"], "Download clock review", "clock-review.csv")
        manual_review(uploaded, report["review"], report["duration_seconds"], lead, tail, state)
