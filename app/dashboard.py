from __future__ import annotations
import tempfile
from pathlib import Path
import streamlit as st
from src.pipeline.realtime_pipeline import RealtimePipeline

st.set_page_config(page_title="Football Vision Lab", page_icon="⚽", layout="wide")
st.title("⚽ Real-Time Football Vision Lab")
st.caption("Prototype: player tracking, pose estimation and inference profiling")
st.warning("Baseline limitation: generic COCO weights are not trained for football broadcast footage. Ball and event detection require sports-specific data and calibration.")

uploaded = st.file_uploader("Upload a short gameplay/broadcast clip", type=["mp4", "mov", "avi"])
with st.sidebar:
    st.header("Inference settings")
    detector_weights = st.text_input("Detection / tracking weights", "yolo11n.pt")
    pose_weights = st.text_input("Pose weights", "yolo11n-pose.pt")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    device = st.text_input("Device (blank = auto)", "")
    max_frames = st.number_input("Maximum frames (0 = all)", min_value=0, value=600, step=100)

if uploaded:
    st.video(uploaded)
    if st.button("Run vision pipeline", type="primary"):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / uploaded.name
            output_path = Path(temp_dir) / "annotated.mp4"
            input_path.write_bytes(uploaded.getbuffer())
            try:
                with st.spinner("Loading models and processing video..."):
                    pipeline = RealtimePipeline(detector_weights, pose_weights,
                                                device=device or None, confidence=confidence)
                    metrics = pipeline.process_video(str(input_path), str(output_path),
                                                      max_frames=max_frames or None)
                    video_bytes = output_path.read_bytes()
                st.success("Processing complete")
                a, b, c = st.columns(3)
                a.metric("Frames processed", metrics["frames_processed"])
                b.metric("Average inference", f"{metrics['average_inference_ms']:.1f} ms/frame")
                c.metric("Inference throughput", f"{metrics['average_inference_fps']:.1f} FPS")
                st.video(video_bytes)
                st.download_button("Download annotated video", video_bytes,
                                   file_name="football_annotated.mp4", mime="video/mp4")
                st.info("FPS shown is model inference throughput, not necessarily end-to-end live display FPS. Benchmark on your target hardware.")
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
else:
    st.info("Upload a short clip to begin. Start with 10–30 seconds while validating your environment.")
