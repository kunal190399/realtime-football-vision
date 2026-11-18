from __future__ import annotations
import csv
import io
import tempfile
from pathlib import Path
import streamlit as st
from src.detection.football_weights import football_weights
from src.events.highlights import export_highlights
from src.events.scoreboard import scan_scoreboard
from src.pipeline.realtime_pipeline import RealtimePipeline

st.set_page_config(page_title="Football Vision Lab", page_icon="⚽", layout="wide")
st.title("⚽ Real-Time Football Vision Lab")
st.caption("Prototype: player tracking, pose estimation and inference profiling")
st.caption("Possession estimates use the distance between a detected ball and nearby players. Review the video to check assignments.")

uploaded = st.file_uploader("Upload a short gameplay/broadcast clip", type=["mp4", "mov", "avi"])
mode = st.radio("Analysis", ["Players and ball", "Scoreboard updates"], horizontal=True)
if mode == "Scoreboard updates":
    st.caption("Preset for the France–Australia score graphic. Reads single-digit scores "
               "in the full video and timestamps changes on the video clock.")
    make_clips = st.checkbox("Export one highlight per score update")
    if make_clips:
        lead = st.slider("Seconds before the score update", 1, 30, 8)
        tail = st.slider("Seconds after the score update", 1, 10, 2)
    if uploaded and st.button("Scan scoreboard", type="primary"):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / ("input" + Path(uploaded.name).suffix)
                path.write_bytes(uploaded.getbuffer())
                with st.spinner("Reading the scoreboard..."):
                    report = scan_scoreboard(str(path))
                bundle = None
                if make_clips and report["events"]:
                    try:
                        with st.spinner("Exporting highlight clips..."):
                            bundle = export_highlights(path, report["events"], report["duration_seconds"],
                                                       lead, tail)
                    except (OSError, RuntimeError, ValueError) as exc:
                        st.warning(f"Highlight export failed: {exc}")
            if bundle:
                st.download_button("Download highlights and index", bundle, file_name="highlights.zip",
                                   mime="application/zip", on_click="ignore")
            st.write(f"{len(report['events'])} score updates. "
                     f"Score readable in {report['readable']} of {report['samples']} samples.")
            if report["events"]:
                st.dataframe(report["events"], hide_index=True)
                buffer = io.StringIO()
                writer = csv.DictWriter(buffer, fieldnames=report["events"][0].keys())
                writer.writeheader()
                writer.writerows(report["events"])
                st.download_button("Download score updates", buffer.getvalue(),
                                   file_name="score-updates.csv", mime="text/csv", on_click="ignore")
            elif not report["readable"]:
                st.info("No readable scores. Check that the video uses the supported scoreboard layout.")
        except Exception as exc:
            st.error(f"Scoreboard scan failed: {exc}")
    st.stop()

with st.sidebar:
    st.header("Inference settings")
    specialist = st.checkbox("Use football-specific detector (experimental)")
    if specialist:
        st.caption("reco YOLO26s · 1920 px · 40 MB on first run. CPU processing can be slow.")
    else:
        detector_weights = st.text_input("Detection / tracking weights", "yolo11n.pt")
    pose_weights = st.text_input("Pose weights", "yolo11n-pose.pt")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    device = st.text_input("Device (blank = auto)", "")
    possession = st.checkbox("Estimate player possession (experimental)")
    max_frames = st.number_input("Maximum frames (0 = all)", min_value=0, value=600, step=100)

if uploaded:
    st.video(uploaded)
    if st.button("Run vision pipeline", type="primary"):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / ("input" + Path(uploaded.name).suffix)
            output_path = Path(temp_dir) / "annotated.mp4"
            input_path.write_bytes(uploaded.getbuffer())
            try:
                with st.spinner("Loading models and processing video..."):
                    if specialist:
                        detector_weights = str(football_weights(Path.home() / ".cache" / "football-vision"))
                    pipeline = RealtimePipeline(detector_weights, pose_weights,
                                                device=device or None, confidence=confidence)
                    metrics = pipeline.process_video(str(input_path), str(output_path),
                                                      max_frames=max_frames or None, possession=possession)
                    video_bytes = output_path.read_bytes()
                st.success("Processing complete")
                a, b, c = st.columns(3)
                a.metric("Frames processed", metrics["frames_processed"])
                b.metric("Average inference", f"{metrics['average_inference_ms']:.1f} ms/frame")
                c.metric("Inference throughput", f"{metrics['average_inference_fps']:.1f} FPS")
                if possession:
                    rows = metrics["timeline"]
                    assigned = sum(row["possession_track_id"] is not None for row in rows)
                    observed = sum(row["ball_source"] in {"tracked", "detected"} for row in rows)
                    estimated = sum(row["ball_source"] == "motion" for row in rows)
                    st.write(f"Ball observed in {observed} of {len(rows)} frames. "
                             f"Motion estimates in {estimated} additional frames.")
                    st.write(f"Possession candidate assigned in {assigned} of {len(rows)} frames.")
                    st.caption("Yellow ring: direct observation. Blue ring: motion estimate.")
                    csv_buffer = io.StringIO()
                    writer = csv.DictWriter(csv_buffer, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                    st.download_button("Download frame timeline", csv_buffer.getvalue(),
                                       file_name="football_timeline.csv", mime="text/csv",
                                       on_click="ignore")
                st.video(video_bytes)
                st.download_button("Download annotated video", video_bytes,
                                   file_name="football_annotated.mp4", mime="video/mp4", on_click="ignore")
                st.info("FPS shown is model inference throughput, not necessarily end-to-end live display FPS. Benchmark on your target hardware.")
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
else:
    st.info("Upload a short clip to begin. Start with 10–30 seconds while validating your environment.")
