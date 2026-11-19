"""Evaluate one clip's visual event timestamps, matching each label at most once."""
import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

from src.evaluation.score_updates import align


@dataclass(order=True)
class Timestamp:
    seconds: float


KINDS = ("shot_candidate", "goal_line_crossing_candidate")


def evaluate(labels, candidates, tolerance=.5):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative.")

    def group(rows):
        groups = {kind: [] for kind in KINDS}
        for row in rows:
            kind, seconds = row["event"], float(row["seconds"])
            if kind not in groups or not math.isfinite(seconds) or seconds < 0:
                raise ValueError(f"Invalid visual event: {row}")
            groups[kind].append(Timestamp(seconds))
        return groups

    expected, observed = group(labels), group(candidates)
    report = {}
    for kind in KINDS:
        pairs, misses, extras = align(sorted(expected[kind]), sorted(observed[kind]), tolerance)
        tp, fn, fp = len(pairs), len(misses), len(extras)
        report[kind] = dict(matches=tp, missed=fn, extra=fp,
                            precision=tp/(tp+fp) if tp+fp else None,
                            recall=tp/(tp+fn) if tp+fn else None,
                            f1=2*tp/(2*tp+fn+fp) if tp+fn+fp else None,
                            timing_errors_seconds=[p.seconds-l.seconds for l, p in reversed(pairs)])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--tolerance", type=float, default=.5)
    args = parser.parse_args()
    with args.labels.open() as labels, args.predictions.open() as predictions:
        print(json.dumps(evaluate(csv.DictReader(labels), csv.DictReader(predictions), args.tolerance), indent=2))
