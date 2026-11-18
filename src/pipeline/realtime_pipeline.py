"""Video inference pipeline: tracking, pose overlay and performance telemetry."""
from __future__ import annotations
import time
from pathlib import Path
import cv2
from src.tracking.ball_motion import BallMotion
from src.tracking.possession import PossessionEstimator
from src.tracking.tracker import PlayerTracker
from src.pose.pose_estimator import PoseEstimator


class RealtimePipeline:
    def __init__(self, detection_weights: str = "yolo11n.pt",
                 pose_weights: str = "yolo11n-pose.pt", device: str | None = None,
                 confidence: float = 0.25) -> None:
        self.tracker = PlayerTracker(detection_weights, confidence, device=device)
        self.pose = PoseEstimator(pose_weights, device=device)

    def process_video(self, input_path: str, output_path: str,
                      max_frames: int | None = None, possession: bool = False) -> dict:
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
            while True:
                ok, frame = cap.read()
                if not ok or (max_frames is not None and count >= max_frames):
                    break
                started = time.perf_counter()
                tracked = self.tracker.track_result(frame, include_ball=possession)
                annotated = self.pose.annotate(frame, canvas=tracked.plot())
                if possession:
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
                elapsed = time.perf_counter() - started
                total_inference += elapsed
                fps = 1.0 / max(elapsed, 1e-9)
                cv2.putText(annotated, f"Inference: {elapsed*1000:.1f} ms | FPS: {fps:.1f}",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                writer.write(annotated)
                count += 1
        finally:
            cap.release()
            writer.release()
        if not count:
            raise ValueError("No video frames could be processed.")
        return {
            "timeline": timeline,
            "frames_processed": count,
            "input_fps": fps_in,
            "average_inference_ms": (total_inference / count * 1000) if count else 0.0,
            "average_inference_fps": (count / total_inference) if total_inference else 0.0,
            "output_path": output_path,
        }
