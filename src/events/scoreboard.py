"""Read score updates from the fixed France–Australia broadcast score graphic."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Callable

import cv2
import numpy as np


@dataclass
class ScoreUpdate:
    frame: int
    seconds: float
    team: str
    previous_score: str
    new_score: str
    confirmed_seconds: float


class ScoreMonitor:
    def __init__(self):
        self.score = None
        self.pending = None
        self.count = 0
        self.last_read = None
        self.start = 0

    def observe(self, score, frame: int, fps: float) -> ScoreUpdate | None:
        if self.last_read is not None and frame - self.last_read > 2 * fps:
            self.score = None
        if score is None:
            self.count = 0
            return None
        self.last_read = frame
        if score != self.pending or self.count == 0:
            self.pending, self.start, self.count = score, frame, 0
        self.count += 1
        if self.count != 3 or score == self.score:
            return None
        previous = self.score
        if previous is not None and (score[0] < previous[0] or score[1] < previous[1]):
            return None
        self.score = score
        if previous is None:
            return None
        change = (score[0] - previous[0], score[1] - previous[1])
        if change not in ((1, 0), (0, 1)):
            return None
        return ScoreUpdate(self.start, self.start / fps, "FRA" if change[0] else "AUS",
                           f"{previous[0]}–{previous[1]}", f"{score[0]}–{score[1]}", frame / fps)


def read_digit(mask: np.ndarray) -> int | None:
    image = cv2.resize(mask, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    image = cv2.copyMakeBorder(image, 15, 15, 20, 20, cv2.BORDER_CONSTANT, value=255)
    result = subprocess.run(
        ["tesseract", "stdin", "stdout", "--psm", "10", "-c", "tessedit_char_whitelist=0123456789"],
        input=cv2.imencode(".png", image)[1].tobytes(), capture_output=True, check=True, timeout=10,
    ).stdout.strip()
    return int(result) if len(result) == 1 and result.isdigit() else None


class ScoreReader:
    def __init__(self, ocr: Callable = read_digit):
        self.ocr = ocr
        self.cache = {}

    def digit(self, crop, side):
        # Check the maroon score cell before interpreting its white glyph.
        b, g, r = cv2.split(crop.astype(np.int16))
        if np.mean((r > g + 45) & (r > b + 35)) < 0.35:
            return None
        white = crop.min(axis=2) > 155
        ys, xs = np.where(white)
        if not len(xs):
            return None
        glyph = white[ys.min():ys.max()+1, xs.min():xs.max()+1]
        if not (10 <= glyph.shape[0] <= 20 and 3 <= glyph.shape[1] <= 16):
            return None
        mask = np.where(glyph, 0, 255).astype(np.uint8)
        cached = self.cache.get(side)
        if cached is not None:
            old, value = cached
            if old.shape == mask.shape and np.mean(old != mask) <= 0.05:
                return value
        digit = self.ocr(mask)
        if digit is None:
            # A slightly brighter threshold can separate compressed digit strokes.
            bright = crop.min(axis=2) > 175
            tighter = bright[ys.min():ys.max()+1, xs.min():xs.max()+1]
            digit = self.ocr(np.where(tighter, 0, 255).astype(np.uint8))
        if digit is not None:
            self.cache[side] = mask, digit
        return digit

    def read(self, frame):
        h, w = frame.shape[:2]
        # Only the top scoreboard band is needed, scaled to the 720p reference.
        band = cv2.resize(frame[:max(1, round(h * 75 / 720))], (1280, 75))
        left = self.digit(band[43:63, 289:307], 0)
        right = self.digit(band[43:63, 358:376], 1)
        return (left, right) if left is not None and right is not None else None


def scan_scoreboard(path: str) -> dict:
    if not shutil.which("tesseract"):
        raise RuntimeError("Install Tesseract OCR and make its tesseract command available on PATH.")
    cap = cv2.VideoCapture(path)
    reader, monitor = ScoreReader(), ScoreMonitor()
    events = []
    frame = samples = readable = 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Could not read the video or its frame rate.")
        stride = max(1, round(fps / 4))
        while cap.grab():
            if frame % stride == 0:
                ok, pixels = cap.retrieve()
                score = reader.read(pixels) if ok else None
                samples += 1
                readable += int(score is not None)
                event = monitor.observe(score, frame, fps)
                if event:
                    events.append(asdict(event))
            frame += 1
        if frame == 0:
            raise ValueError("No decodable frames.")
        return dict(events=events, samples=samples, readable=readable, frames=frame,
                    duration_seconds=frame / fps)
    finally:
        cap.release()
