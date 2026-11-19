"""Fit the small-ball appearance model using the training image list."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from src.detection.local_ball import features


def examples(path):
    image = cv2.imread(str(path))
    xy, values, scale = features(image)
    h, w = image.shape[:2]
    boxes = []
    label_path = path.parent.parent / "labels" / path.with_suffix(".txt").name
    for line in label_path.read_text().splitlines():
        cls, x, y, width, height = map(float, line.split())
        if cls == 0:
            boxes.append([x*w*scale, y*h*scale, width*w*scale, height*h*scale])
    boxes = np.asarray(boxes).reshape(-1, 4)
    if len(boxes) and len(xy):
        distances = np.linalg.norm(xy[:, None, :] - boxes[None, :, :2], axis=2)
        closest = boxes[distances.argmin(axis=1)].copy()
        closest[:, :2] -= xy
        return values, distances.min(axis=1), closest
    return values, np.full(len(xy), np.inf), np.zeros((len(xy), 4))


def train(paths, output):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(42)
    positive, negative, offsets = [], [], []
    for path in paths:
        values, distance, boxes = examples(path)
        positive.append(values[distance <= 4.5])
        offsets.append(boxes[distance <= 4.5])
        far = np.flatnonzero(distance > 12)
        negative.append(values[rng.choice(far, min(190, len(far)), replace=False)])
    pos = np.concatenate(positive)
    if len(pos) == 0:
        raise ValueError("No ball candidates near training labels")
    model = HistGradientBoostingClassifier(max_iter=85, max_leaf_nodes=12, min_samples_leaf=12,
                                            l2_regularization=1, random_state=42)
    for iteration in range(2):
        neg = np.concatenate(negative)
        model.fit(np.concatenate([pos, neg]), np.r_[np.ones(len(pos)), np.zeros(len(neg))])
        if iteration == 0:
            for path in paths:
                values, distance, _ = examples(path)
                far = values[distance > 12]
                if len(far):
                    scores = model.predict_proba(far)[:, 1]
                    negative.append(far[np.argsort(scores)[-8:]])
    reg = Ridge(alpha=1e6).fit(pos, np.concatenate(offsets))
    # Export only the numeric split and leaf arrays. Inference uses NumPy.
    trees = [stage[0].nodes for stage in model._predictors]
    fields = ("feature_idx", "num_threshold", "left", "right", "value", "is_leaf")
    arrays = {field: np.stack([np.pad(t[field], (0, max(map(len, trees))-len(t))) for t in trees])
              for field in fields}
    np.savez_compressed(output, **arrays, bias=model._baseline_prediction.ravel(),
                        coef=reg.coef_, intercept=reg.intercept_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_list", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    train([Path(s) for s in args.train_list.read_text().splitlines()], args.output)
