"""Train a football detector with source-group splits and evaluate its best checkpoint."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import yaml


def prepare(dataset, output):
    groups = defaultdict(list)
    for image in sorted(dataset.glob("*/images/*.jpg")):
        label = image.parent.parent / "labels" / (image.stem + ".txt")
        if not label.is_file():
            raise ValueError(f"Missing label file: {label}")
        with image.open("rb") as stream:
            if stream.read(2) != b"\xff\xd8":
                raise ValueError(f"Invalid JPEG: {image}. Fetch Git LFS objects before training.")
        groups[image.name.split("_")[0]].append(image.resolve())
    if len(groups) < 5:
        raise ValueError("Expected at least five source groups.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new output directory for this experiment.")
    output.mkdir(parents=True, exist_ok=True)
    config = {"path": str(output.resolve()), "names": yaml.safe_load((dataset/"data.yaml").read_text())["names"]}
    splits = {"train": [], "val": [], "test": []}
    for i, group in enumerate(sorted(groups)):
        splits[{0: "test", 1: "val"}.get(i % 8, "train")].extend(groups[group])
    for name, images in splits.items():
        path = output / f"{name}.txt"
        path.write_text("".join(f"{image}\n" for image in images))
        config[name] = path.name
    path = output / "dataset.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def main():
    from ultralytics import YOLO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    data = prepare(args.dataset, args.output)
    common = dict(data=str(data.resolve()), imgsz=args.imgsz, batch=args.batch, device=args.device,
                  workers=0, project=str(args.output.resolve()), plots=True)
    model = YOLO(args.weights)
    model.train(**common, epochs=args.epochs, name="train", seed=42, mosaic=0,
                amp=False, optimizer="AdamW", lr0=.001, freeze=10)
    candidate = YOLO(str(model.trainer.best))
    metrics = candidate.val(**common, split="test", name="held-out")
    report = dict(weights="train/weights/best.pt", split_counts={
        k: len((args.output / f"{k}.txt").read_text().splitlines()) for k in ("train", "val", "test")},
        overall=metrics.results_dict, per_class=metrics.summary())
    (args.output / "test-metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
