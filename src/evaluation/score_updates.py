"""Compare dashboard CSV exports with labels for one source video."""
import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from statistics import mean


@dataclass(frozen=True)
class Change:
    seconds: float
    team: str
    new_score: str


def load_changes(path: Path) -> list[Change]:
    changes = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"seconds", "team", "new_score"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: expected seconds, team and new_score columns")
        for row in reader:
            try:
                seconds = float(row["seconds"])
                team = (row["team"] or "").strip().upper()
                score = (row["new_score"] or "").strip().replace("–", "-")
                if not math.isfinite(seconds) or seconds < 0 or not team or not re.fullmatch(r"\d+-\d+", score):
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError(f"{path}, line {reader.line_num}: invalid time, team or score") from None
            changes.append(Change(seconds, team, score))
    return changes


def align(expected, observed, tolerance):
    # Within one team/score group, maximise pair count, then minimise total absolute error.
    height, width = len(expected), len(observed)
    best = [[(0, 0.0)] * (width + 1) for _ in range(height + 1)]
    moves = {}
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            choices = [(best[i-1][j], "miss"), (best[i][j-1], "extra")]
            delta = abs(expected[i-1].seconds - observed[j-1].seconds)
            if delta <= tolerance + 1e-9:
                count, loss = best[i-1][j-1]
                choices.append(((count+1, loss-delta), "pair"))
            best[i][j], moves[i, j] = max(choices, key=lambda entry: entry[0])
    pairs, missed, extra = [], [], []
    i, j = height, width
    while i or j:
        move = moves.get((i, j), "miss" if i else "extra")
        if move == "pair":
            pairs.append((expected[i-1], observed[j-1]))
        elif move == "miss":
            missed.append(expected[i-1])
        else:
            extra.append(observed[j-1])
        if move != "extra":
            i -= 1
        if move != "miss":
            j -= 1
    return pairs, missed, extra


def evaluate(labels, predictions, tolerance):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and non-negative")
    matches, missing, extra = [], [], []
    groups = {(r.team, r.new_score) for r in [*labels, *predictions]}
    for team, score in sorted(groups):
        def group(rows):
            return sorted((r for r in rows if (r.team, r.new_score) == (team, score)), key=lambda r: r.seconds)
        paired, missed, extras = align(group(labels), group(predictions), tolerance)
        for truth, pred in paired:
            matches.append(dict(**asdict(truth), predicted_seconds=pred.seconds,
                                error_seconds=round(pred.seconds-truth.seconds, 6)))
        missing.extend(asdict(r) for r in missed)
        extra.extend(asdict(r) for r in extras)
    for rows in (matches, missing, extra):
        rows.sort(key=lambda r: r["seconds"])
    tp, fp, fn = len(matches), len(extra), len(missing)
    return dict(tolerance_seconds=tolerance, matched=tp, extra=fp, missed=fn,
                precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                mean_absolute_error_seconds=mean(abs(m["error_seconds"]) for m in matches) if tp else None,
                matches=matches, missed_labels=missing, extra_predictions=extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("predictions", type=Path, nargs="*", help="CSV exports from the same clip (omit for zero predictions)")
    parser.add_argument("--tolerance", type=float, default=0.25, help="Maximum timing error in seconds")
    args = parser.parse_args()
    try:
        predictions = [row for path in args.predictions for row in load_changes(path)]
        result = evaluate(load_changes(args.labels), predictions, args.tolerance)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
