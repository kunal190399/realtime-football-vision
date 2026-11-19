"""Generate a six-second schematic clip and check the scoreboard and event workflows."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import tempfile
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.events.goal_mouth import GoalMonitor, GoalMouth
from src.events.scoreboard import ScoreLayout, scan_scoreboard
from src.events.highlights import export_highlights
from src.evaluation.visual_events import evaluate


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    goal = GoalMouth((1080, 280), (1080, 440), (640, 360))
    layout = ScoreLayout(("RED", "BLU"), ((263, 26, 26, 33), (332, 26, 26, 33)))
    monitor = GoalMonitor(goal)
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    predictions = []
    clip = output / "fixed-camera-demo.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        raw = str(Path(tmp) / "render.avi")
        writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"MJPG"), 25, (1280, 720))
        if not writer.isOpened():
            raise RuntimeError("Cannot open MJPEG encoder")
        try:
            for frame in range(150):
                image = np.full((720, 1280, 3), (60, 130, 53), np.uint8)
                cv2.rectangle(image, (80, 110), (1080, 610), (235, 235, 235), 3)
                cv2.rectangle(image, (1080, 280), (1120, 440), (210, 210, 210), 3)
                for player, colour in [((585, 369), (40, 40, 230)), ((875, 415), (220, 75, 35))]:
                    cv2.circle(image, player, 16, colour, -1, cv2.LINE_AA)
                ball = (min(1120, 610 + max(0, frame-25)*10), 360)
                cv2.circle(image, ball, 7, (255, 255, 255), -1, cv2.LINE_AA)
                board = Image.new("RGB", (500, 46), (80, 18, 32))
                draw = ImageDraw.Draw(board)
                for x, value in [(8, f"12:0{frame//25}"), (128, "RED"), (226, "1" if frame<85 else "2"),
                                 (295, "1"), (373, "BLU")]:
                    draw.text((x, 1), value, font=font, fill="white")
                image[24:70, 40:540] = np.array(board)[:, :, ::-1]
                writer.write(image)
                predictions.extend(monitor.observe(frame/25, ball, "tracked", 1))
        finally:
            writer.release()
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", raw, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(clip)], check=True)
    labels = [dict(event="shot_candidate", seconds=1.04), dict(event="goal_line_crossing_candidate", seconds=2.88)]
    for name, rows in [("reference", labels), ("candidates", predictions)]:
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["event", "seconds"])
            writer.writeheader()
            writer.writerows(rows)
    geometry = evaluate(labels, predictions)
    scan = scan_scoreboard(str(clip), layout, (43, 26, 100, 36))
    report = dict(scripted_trajectory=geometry, scoreboard=scan)
    (output/"demo-results.json").write_text(json.dumps(report, indent=2))
    assert all(row["f1"] == 1 for row in geometry.values()), geometry
    assert len(scan["events"]) == 1 and not scan["review"], scan
    event = scan["events"][0]
    assert event["new_score"] == "2–1" and abs(event["seconds"] - 3.4) <= .25, event
    archive = output / "highlight.zip"
    archive.write_bytes(export_highlights(clip, scan["events"], scan["duration_seconds"], 3, 1))
    with ZipFile(archive) as result:
        assert result.testzip() is None and any(name.endswith(".mp4") for name in result.namelist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
