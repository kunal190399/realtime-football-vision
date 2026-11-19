"""Export source-video excerpts around confirmed scoreboard changes."""
import csv
import io
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile


def clip_window(seconds: float, duration: float, lead: float, tail: float):
    if not (0 <= seconds < duration and lead >= 0 and tail > 0):
        raise ValueError("Invalid event time or highlight window.")
    return max(0.0, seconds - lead), min(duration, seconds + tail)


def export_highlights(source: Path, events: list[dict], duration: float,
                      lead: float = 8.0, tail: float = 2.0) -> bytes:
    if not events:
        raise ValueError("At least one confirmed score change is required.")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Install FFmpeg and make ffmpeg available on PATH.")
    manifest = io.StringIO()
    writer = csv.writer(manifest)
    reviewed = any(event.get("review_reason") for event in events)
    audit_fields = ["selection", "review_reason", "previous_clock", "new_clock", "previous_sample_seconds"]
    writer.writerow(["file", "team", "score", "source_start_s", "source_end_s",
                     "score_update_s", "update_in_clip_s"] + (audit_fields if reviewed else []))
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        archive = folder / "highlights.zip"
        with ZipFile(archive, "w") as bundle:
            for index, event in enumerate(events, 1):
                seconds = event["seconds"]
                start, end = clip_window(seconds, duration, event.get("lead_seconds", lead),
                                         event.get("tail_seconds", tail))
                team = event["team"]
                score = event["new_score"]
                name = f"{index:02d}-{team}-{score.replace('–', '-')}.mp4"
                clip = folder / "clip.mp4"
                command = [
                    "ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start:.6f}",
                    "-i", str(source), "-t", f"{end-start:.6f}", "-map", "0:v:0",
                    "-map", "0:a:0?", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "19", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:a", "aac", "-movflags", "+faststart", str(clip)
                ]
                result = subprocess.run(command, capture_output=True)
                if result.returncode:
                    raise RuntimeError(f"Could not encode highlight {index}: "
                                       + result.stderr.decode(errors="replace")[-600:])
                bundle.write(clip, name)
                audit = []
                if reviewed:
                    audit = ["manual" if event.get("review_reason") else "automatic",
                             *(event.get(field, "") for field in audit_fields[1:])]
                writer.writerow([name, team, score, round(start, 4), round(end, 4),
                                 round(seconds, 4), round(seconds-start, 4)] + audit)
            bundle.writestr("highlights.csv", manifest.getvalue())
        return archive.read_bytes()
