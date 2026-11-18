"""Video inference pipeline: tracking, pose overlay and performance telemetry."""
from __future__ import annotations
import time
from pathlib import Path
import cv2
from src.evaluation.runtime import RuntimeProfile
from src.tracking.ball_motion import BallMotion
from src.tracking.possession import PossessionEstimator
from src.tracking.tracker import PlayerTracker
from src.pose.pose_estimator import PoseEstimator


class RealtimePipeline:
    def __init__(self, detection_weights: str = "yolo11n.pt",
                 pose_weights: str = "yolo11n-pose.pt", device: str | None = None,
                 confidence: float = 0.25) -> None:
        self.settings = dict(mode="vision", detector=Path(detection_weights).name,
                             pose=Path(pose_weights).name, requested_device=device or "auto", confidence=confidence)
        self.tracker = PlayerTracker(detection_weights, confidence, device=device)
        self.pose = PoseEstimator(pose_weights, device=device)

    def process_video(self, input_path: str, output_path: str,
                      max_frames: int | None = None, possession: bool = False) -> dict:
        runtime = RuntimeProfile()
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_in, (width, height))
        count, total_inference = 0, 0.0
        estimator = PossessionEstimator()
        motion = BallMotion(fps_in)
        timeline = []
        try:
            if not writer.isOpened():
                raise ValueError("Could not open the video encoder.")
            while max_frames is None or count < max_frames:
                started = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break
                runtime.record("Video decode", started)
                started = time.perf_counter()
                tracked = self.tracker.track_result(frame, include_ball=possession)
                runtime.record("Detection and tracking", started)
                stage = time.perf_counter()
                annotated = self.pose.annotate(frame, canvas=tracked.plot())
                runtime.record("Pose and model overlays", stage)
                if possession:
                    stage = time.perf_counter()
                    observation = estimator.update(self.tracker.observations(tracked))
                    motion.annotate_observation(frame, observation)
                    source = observation["ball_source"]
                    timeline.append(dict(frame=count, seconds=count / fps_in, **observation))
                    if observation["ball_x"] is not None:
                        centre = (round(observation["ball_x"]), round(observation["ball_y"]))
                        colour = (255, 180, 0) if source == "motion" else (0, 240, 255)
                        cv2.circle(annotated, centre, 15, colour, 2, cv2.LINE_AA)
                    owner = observation["possession_track_id"]
                    label = f"track {owner}" if owner is not None else "unknown"
                    cv2.putText(annotated, f"Possession candidate: {label}", (20, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    runtime.record("Possession and overlay", stage)
                elapsed = time.perf_counter() - started
                total_inference += elapsed
                fps = 1.0 / max(elapsed, 1e-9)
                cv2.putText(annotated, f"Inference: {elapsed*1000:.1f} ms | FPS: {fps:.1f}",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                stage = time.perf_counter()
                writer.write(annotated)
                runtime.record("Video write", stage)
                count += 1
        finally:
            cap.release()
            writer.release()
        if not count:
            raise ValueError("No video frames could be processed.")
        return {
            "runtime": runtime.report(count, fps_in, width, height,
                                      dict(self.settings, possession=possession, max_frames=max_frames)),
            "timeline": timeline,
            "frames_processed": count,
            "input_fps": fps_in,
            "average_inference_ms": (total_inference / count * 1000) if count else 0.0,
            "average_inference_fps": (count / total_inference) if total_inference else 0.0,
            "output_path": output_path,
        }
