"""Check the match clock around stable, one-goal scoreboard increments."""
from itertools import groupby
import re
import subprocess

import cv2
import numpy as np


def clock_seconds(text):
    match = re.fullmatch(r"(\d{1,3}):([0-5]\d)", text or "")
    if match:
        return 60 * int(match[1]) + int(match[2])
    return None


def read_clock(cap, frame, box, light_digits):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = cap.read()
    if not ok:
        return None
    x, y, w, h = box
    crop = cv2.resize(image, (1280, 720))[y:y+h, x:x+w]
    ink = crop if light_digits else 255 - crop
    mask = (ink.min(axis=2) > 150).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(mask)
    if not h or w < h:
        return None
    glyphs = 255 - mask[y:y+h, x:x+w] * 255
    scale = min(4, 640 / w, 96 / h)
    scaled = cv2.resize(glyphs, None, fx=scale, fy=scale)
    padded = cv2.copyMakeBorder(scaled, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    text = subprocess.run(
        ["tesseract", "stdin", "stdout", "--psm", "7", "-c", "tessedit_char_whitelist=0123456789:"],
        input=cv2.imencode(".png", padded)[1].tobytes(), capture_output=True,
        check=True, timeout=10,
    ).stdout.decode().strip()
    return text if clock_seconds(text) is not None else None


def review_changes(samples, fps, teams, events, clock_at):
    by_frame = {event["frame"]: event for event in events}
    previous = None
    accepted, review = [], []
    for score, group in groupby(samples, key=lambda sample: sample[1]):
        frames = [frame for frame, _ in group]
        if score is None or len(frames) < 3:
            continue
        if previous is not None:
            old, last = previous
            diff = (score[0]-old[0], score[1]-old[1])
            if diff in ((1, 0), (0, 1)):
                first = frames[0]
                gap = (first-last)/fps
                before, after = clock_at(last), clock_at(first)
                a, b = clock_seconds(before), clock_seconds(after)
                reasons = []
                if gap > 2:
                    reasons.append("scoreboard gap")
                if a is None or b is None:
                    reasons.append("unreadable clock")
                elif b < a or abs((b-a)-gap) > 2:
                    reasons.append("match-clock discontinuity")
                event = by_frame.get(first)
                if event is None and not reasons:
                    reasons.append("score continuity unconfirmed")
                row = dict(frame=first, seconds=first/fps, team=teams[0 if diff[0] else 1],
                           previous_score=f"{old[0]}–{old[1]}", new_score=f"{score[0]}–{score[1]}",
                           confirmed_seconds=frames[2]/fps, previous_sample_seconds=last/fps,
                           previous_clock=before, new_clock=after, review_reason=", ".join(reasons))
                (review if reasons else accepted).append(row)
        previous = score, frames[-1]
    return accepted, review
