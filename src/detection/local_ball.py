"""Find small bright ball candidates in a 1080p reference view."""
from pathlib import Path

import cv2
import numpy as np

from src.tracking.observations import Track


def features(frame):
    height, width = frame.shape[:2]
    scale = 1080 / height
    image = cv2.resize(frame, (round(width * scale), 1080))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fine = cv2.GaussianBlur(gray, (0, 0), 1.8)
    coarse = cv2.GaussianBlur(gray, (0, 0), 6)
    contrast = cv2.subtract(fine, coarse)
    peaks = (contrast == cv2.dilate(contrast, np.ones((9, 9), np.uint8))) & (contrast >= 15)
    _, _, _, centres = cv2.connectedComponentsWithStats(peaks.astype(np.uint8))
    xy = centres[1:].round().astype(int)
    keep = ((xy[:, 0] >= 24) & (xy[:, 0] < image.shape[1]-24) &
            (xy[:, 1] >= 24) & (xy[:, 1] < image.shape[0]-24))
    xy = xy[keep]
    x, y = xy.T
    # Sample colour and local appearance at the same physical scale for every candidate.
    offsets = np.arange(-18, 19, 3)
    yy = y[:, None, None] + offsets[None, :, None]
    xx = x[:, None, None] + offsets[None, None, :]
    patch = fine[yy, xx].astype(np.float32)
    relative = patch - coarse[y, x, None, None].astype(np.float32)
    colors = image[yy[:, ::2], xx[:, :, ::2]].reshape(len(xy), 147)
    values = np.column_stack([relative.reshape(len(xy), 169), colors]).astype(np.float32)
    return xy, values, scale


class LocalBallDetector:
    def __init__(self, path=None):
        if path is None:
            path = Path(__file__).resolve().parents[2] / "configs" / "local-ball.npz"
        with np.load(path, allow_pickle=False) as archive:
            self.model = dict(archive)

    def detect(self, frame, threshold=0.90):
        candidate = self.strongest(frame)
        return candidate if candidate is not None and candidate.confidence >= threshold else None

    def strongest(self, frame):
        xy, values, scale = features(frame)
        if not len(xy):
            return None
        model = self.model
        scores = np.full(len(xy), model["bias"][0])
        rows = np.arange(len(xy))
        for tree, leaves in enumerate(model["is_leaf"]):
            nodes = np.zeros(len(rows), dtype=int)
            active = rows
            while len(active):
                node = nodes[active]
                right = values[active, model["feature_idx"][tree, node]] > model["num_threshold"][tree, node]
                nodes[active] = np.where(right, model["right"][tree, node], model["left"][tree, node])
                active = active[~leaves[nodes[active]].astype(bool)]
            scores += model["value"][tree, nodes]
        best = scores.argmax()
        probability = float(1 / (1 + np.exp(-scores[best])))
        dx, dy, width, height = model["coef"] @ values[best] + model["intercept"]
        cx, cy = xy[best] + np.clip([dx, dy], -4, 4)
        half_w, half_h = np.clip([width, height], 6, 22) / 2
        bounds = np.array([cx-half_w, cy-half_h, cx+half_w, cy+half_h]) / scale
        return Track("ball", None, tuple(bounds.tolist()), probability)
