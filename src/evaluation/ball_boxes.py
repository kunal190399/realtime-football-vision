"""Check deployed detector ball boxes against YOLO labels at one operating point."""
import argparse
import json
from pathlib import Path

import cv2

from src.detection.football_weights import football_weights


def intersection_over_union(first, second):
    a, b, c, d = first
    x, y, z, t = second
    intersection = max(0, min(z, c)-max(x, a)) * max(0, min(t, d)-max(y, b))
    union = (c-a)*(d-b) + (z-x)*(t-y) - intersection
    return intersection / union if union > 0 else 0


def detector_boxes(model, frame, confidence, imgsz):
    result = model.predict(frame, device="cpu", imgsz=imgsz, conf=confidence, verbose=False)[0]
    return [box.xyxy[0].tolist() for box in sorted(result.boxes, key=lambda b: -b.conf.item())
            if result.names[int(box.cls.item())].lower() in {"ball", "football", "sports ball"}]


def evaluate(model, paths, confidence, imgsz, local=False):
    matched = extra = missed = 0
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Unreadable image: {path}")
        height, width = frame.shape[:2]
        labels = []
        label_file = path.parent.parent / "labels" / (path.stem+".txt")
        for row in label_file.read_text().splitlines():
            cls, x, y, w, h = map(float, row.split())
            if cls == 0:  # Ball in the pinned four-class football dataset.
                labels.append(((x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height))
        if local:
            candidate = model.detect(frame, confidence)
            boxes = [candidate.xyxy] if candidate is not None else []
        else:
            boxes = detector_boxes(model, frame, confidence, imgsz)
        for box in boxes:
            scores = [intersection_over_union(box, label) for label in labels]
            if scores and max(scores) >= .5:
                matched += 1
                labels.pop(scores.index(max(scores)))
            else:
                extra += 1
        missed += len(labels)
    return dict(images=len(paths), confidence=confidence, iou=.5, true_positives=matched,
                false_positives=extra, false_negatives=missed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_list", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--local-ball", action="store_true", help="Evaluate the bundled appearance model.")
    selection.add_argument("--weights", type=Path, help="Defaults to the pinned dashboard preset.")
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--confidence", type=float)
    args = parser.parse_args()
    if args.local_ball:
        from src.detection.local_ball import LocalBallDetector
        model = LocalBallDetector()
        default_confidence = .90
    else:
        from ultralytics import YOLO
        model = YOLO(str(args.weights or football_weights(Path.home()/".cache"/"football-vision")), task="detect")
        default_confidence = .25
    confidence = args.confidence if args.confidence is not None else default_confidence
    paths = [Path(s) for s in args.image_list.read_text().splitlines()]
    print(json.dumps(evaluate(model, paths, confidence, args.imgsz, args.local_ball), indent=2))
