"""Export stationary-camera motion candidates with the dashboard's geometry and timing."""
import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np

from src.events.goal_motion import GoalMotion
from src.events.goal_mouth import GoalMouth


def scan(path, goal):
    cap = cv2.VideoCapture(str(path))
    monitor = GoalMotion(goal)
    events = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Cannot read video or frame rate")
        frame = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            events.extend(monitor.observe(image, frame/fps))
            frame += 1
        if not frame:
            raise ValueError("No decodable frames")
    finally:
        cap.release()
    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    for point in ("first", "second", "field"):
        parser.add_argument("--"+point, type=int, nargs=2, metavar=("X", "Y"), required=True)
    args = parser.parse_args()
    events = scan(args.video, GoalMouth(tuple(args.first), tuple(args.second), tuple(args.field)))
    writer = csv.DictWriter(sys.stdout, fieldnames=["event", "seconds"])
    writer.writeheader()
    writer.writerows(events)
