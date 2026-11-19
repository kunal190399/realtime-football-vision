from __future__ import annotations
from src.events.goal_mouth import GoalMouth
import csv
import io
import tempfile
from pathlib import Path
import streamlit as st
from app.runtime_report import show_runtime
from app.score_calibration import select_layout
from src.detection.football_weights import football_weights
from app.score_review import download_rows, show_score_scan

st.set_page_config(page_title="Football Vision Lab", page_icon="⚽", layout="wide")
st.title("⚽ Real-Time Football Vision Lab")
st.caption("Prototype: player tracking, pose estimation and inference profiling")
st.caption("Possession estimates use the distance between a detected ball and nearby players. Review the video to check assignments.")

uploaded = st.file_uploader("Upload a short gameplay/broadcast clip", type=["mp4", "mov", "avi", "webm", "ogv"])
mode = st.radio("Analysis", ["Players and ball", "Scoreboard updates"], horizontal=True)
if mode == "Scoreboard updates":
    preset = st.selectbox("Scoreboard layout", ["France–Australia", "Custom layout"])
    layout = None
    if preset == "Custom layout":
        layout = select_layout(uploaded)
        if layout is None:
            st.stop()
    else:
        st.caption("Reads the France–Australia single-digit score graphic.")
    clock_box = None
    if st.checkbox("Check match-clock continuity before exporting highlights", value=True):
        raw_box = st.text_input("Clock box (x, y, width, height)", "64,40,98,25") if layout else "64,40,98,25"
        try:
            clock_box = tuple(int(value.strip()) for value in raw_box.split(","))
            if len(clock_box) != 4:
                raise ValueError
        except ValueError:
            st.warning("Enter four comma-separated integers for the match clock.")
            st.stop()
    make_clips = st.checkbox("Export one highlight per score update")
    lead, tail = 8, 2
    if make_clips:
        lead = st.slider("Seconds before the score update", 1, 30, 8)
        tail = st.slider("Seconds after the score update", 1, 10, 2)
    try:
        show_score_scan(uploaded, layout, clock_box, make_clips, lead, tail)
    except Exception as exc:
        st.error(f"Scoreboard workflow failed: {exc}")
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
    goal = None
    goal_motion = False
    if st.checkbox("Check shot and crossing candidates (fixed camera)"):
        st.caption("Enter the base of each post and a point on the field in 1280 × 720 reference coordinates. Use one stationary view throughout.")
        try:
            points = [tuple(map(int, st.text_input(label, initial).split(","))) for label, initial in
                      (("First post", "1080,280"), ("Second post", "1080,440"), ("Field point", "640,360"))]
            goal = GoalMouth(*points)
            goal_motion = st.checkbox("Use motion near the goal for event timing",
                                      help="Follows small bright objects moving towards the posts in a stationary view.")
        except ValueError as exc:
            st.warning(str(exc))
            st.stop()
    local_ball = False
    if possession or (goal and not goal_motion):
        local_ball = st.checkbox("Recover small balls with the appearance model", value=False,
                                 help="Checks frames with no detector ball box. White socks can resemble a ball.")
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
                    from src.pipeline.realtime_pipeline import RealtimePipeline

                    pipeline = RealtimePipeline(detector_weights, pose_weights,
                                                device=device or None, confidence=confidence)
                    metrics = pipeline.process_video(str(input_path), str(output_path),
                                                      max_frames=max_frames or None, possession=possession, goal=goal, local_ball=local_ball,
                                                      goal_motion=goal_motion)
                    video_bytes = output_path.read_bytes()
                st.success("Processing complete")
                show_runtime(metrics["runtime"])
                a, b, c = st.columns(3)
                a.metric("Frames processed", metrics["frames_processed"])
                b.metric("Average inference", f"{metrics['average_inference_ms']:.1f} ms/frame")
                c.metric("Inference throughput", f"{metrics['average_inference_fps']:.1f} FPS")
                if goal:
                    events = metrics["goal_events"]
                    st.write(f"{len(events)} visual event candidates. Review the projected ball path in the annotated video.")
                    if events:
                        st.dataframe(events, hide_index=True)
                        download_rows(events, "Download visual candidates", "goal-candidates.csv")
                if possession:
                    rows = metrics["timeline"]
                    assigned = sum(row["possession_track_id"] is not None for row in rows)
                    observed = sum(row["ball_source"] in {"tracked", "detected", "appearance"} for row in rows)
                    estimated = sum(row["ball_source"] == "motion" for row in rows)
                    st.write(f"Ball observed in {observed} of {len(rows)} frames. "
                             f"Motion estimates in {estimated} additional frames.")
                    st.write(f"Possession candidate assigned in {assigned} of {len(rows)} frames.")
                    if local_ball:
                        recovered = sum(row["ball_source"] == "appearance" for row in rows)
                        st.write(f"Appearance model supplied {recovered} direct observations.")
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
