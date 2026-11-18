"""Read score updates from fixed-position broadcast score graphics."""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class ScoreLayout:
    teams: tuple[str, str]
    boxes: tuple[tuple[int, int, int, int], tuple[int, int, int, int]]
    light_digits: bool = True

    def __post_init__(self):
        if len(self.teams) != 2 or len(set(self.teams)) != 2 or not all(
            re.fullmatch(r"[A-Z0-9]{1,8}", team) for team in self.teams
        ):
            raise ValueError("Enter two distinct team codes of 1–8 letters or digits.")
        if len(self.boxes) != 2:
            raise ValueError("Provide one score box for each team.")
        for x, y, width, height in self.boxes:
            if min(x, y) < 0 or min(width, height) < 3 or x + width > 1280 or y + height > 720:
                raise ValueError("Score boxes must fit inside the 1280 × 720 preview.")
        (x, y, w, h), (a, b, c, d) = self.boxes
        if x < a+c and a < x+w and y < b+d and b < y+h:
            raise ValueError("The two score boxes overlap.")


@dataclass
class ScoreUpdate:
    frame: int
    seconds: float
    team: str
    previous_score: str
    new_score: str
    confirmed_seconds: float


class ScoreMonitor:
    def __init__(self, teams=("FRA", "AUS")):
        self.teams = teams
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
        return ScoreUpdate(self.start, self.start / fps, self.teams[0] if change[0] else self.teams[1],
                           f"{previous[0]}–{previous[1]}", f"{score[0]}–{score[1]}", frame / fps)


def ocr_character(image, mode, alphabet) -> bytes:
    result = subprocess.run(
        ["tesseract", "stdin", "stdout", "--psm", mode, "-c", f"tessedit_char_whitelist={alphabet}"],
        input=cv2.imencode(".png", image)[1].tobytes(), capture_output=True, check=True, timeout=10,
    ).stdout.strip()
    return result


def read_digit(mask: np.ndarray) -> int | None:
    image = cv2.resize(mask, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    image = cv2.copyMakeBorder(image, 15, 15, 20, 20, cv2.BORDER_CONSTANT, value=255)
    result = ocr_character(image, "10", "0123456789")
    return int(result) if len(result) == 1 and result.isdigit() else None


class ScoreReader:
    def __init__(self, ocr: Callable = read_digit, layout: ScoreLayout | None = None):
        self.layout = layout
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

    def custom_digit(self, cell):
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        if int(gray.max()) - int(gray.min()) < 45:
            return None
        polarity = cv2.THRESH_BINARY if self.layout.light_digits else cv2.THRESH_BINARY_INV
        _, foreground = cv2.threshold(gray, 0, 255, polarity | cv2.THRESH_OTSU)
        x, y, w, h = cv2.boundingRect(foreground)
        if not w or h < cell.shape[0] * 0.35 or cv2.countNonZero(foreground) > cell.shape[0]*cell.shape[1]*0.65:
            return None
        glyph = 255 - foreground[y:y+h, x:x+w]
        key = (glyph.shape, glyph.tobytes())
        if key not in self.cache:
            if len(self.cache) >= 256:
                self.cache.clear()
            scaled = cv2.resize(glyph, (min(128, max(1, round(w * 64 / h))), 64))
            image = cv2.copyMakeBorder(scaled, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
            char = ocr_character(image, "10", "0123456789")
            if not char:
                char = ocr_character(image, "13", "0123456789Oo")
            if char in (b"O", b"o"):
                char = b"0"
            digit = int(char) if len(char) == 1 and char.isdigit() else None
            if digit is None:
                # Compressed, neutral-coloured digits can separate better by brightness.
                contrast = cell if self.layout.light_digits else 255 - cell
                for level in (155, 175):
                    bright = contrast.min(axis=2) > level
                    ys, xs = np.where(bright)
                    if not len(xs):
                        continue
                    fallback = bright[ys.min():ys.max()+1, xs.min():xs.max()+1]
                    normalized = cv2.resize(fallback.astype(np.uint8),
                                            (max(1, round(fallback.shape[1]*15/fallback.shape[0])), 15),
                                            interpolation=cv2.INTER_NEAREST)
                    digit = self.ocr(255 - normalized * 255)
                    if digit is not None:
                        break
            if digit is None:
                return None
            self.cache[key] = digit
        return self.cache[key]

    def read(self, frame):
        if self.layout is not None:
            preview = cv2.resize(frame, (1280, 720))
            pair = tuple(self.custom_digit(preview[y:y+h, x:x+w])
                         for x, y, w, h in self.layout.boxes)
            return pair if None not in pair else None
        h, w = frame.shape[:2]
        # Only the top scoreboard band is needed, scaled to the 720p reference.
        band = cv2.resize(frame[:max(1, round(h * 75 / 720))], (1280, 75))
        left = self.digit(band[43:63, 289:307], 0)
        right = self.digit(band[43:63, 358:376], 1)
        return (left, right) if left is not None and right is not None else None


def scan_scoreboard(path: str, layout: ScoreLayout | None = None) -> dict:
    if not shutil.which("tesseract"):
        raise RuntimeError("Install Tesseract OCR and make its tesseract command available on PATH.")
    cap = cv2.VideoCapture(path)
    reader, monitor = ScoreReader(layout=layout), ScoreMonitor(layout.teams if layout else ("FRA", "AUS"))
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
