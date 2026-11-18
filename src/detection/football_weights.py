"""Fetch the pinned football detector for the optional dashboard preset."""
from hashlib import sha256
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


REVISION = "d20231484951d4ebf3f9f2a7437aa922a4c1b395"
URL = f"https://huggingface.co/Dura-S/reco-yolo26s-football/resolve/{REVISION}/best.onnx"
SHA256 = "8f9f94c258890ce739a6c2bbf9787827c571ce26796dd600d9c3e4cfb552f0d4"


def football_weights(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"reco-football-{REVISION[:12]}.onnx"
    if path.is_file() and digest(path) == SHA256:
        return path
    with NamedTemporaryFile(dir=cache, delete=False) as temp:
        partial = Path(temp.name)
    try:
        with urlopen(URL, timeout=60) as response, partial.open("wb") as output:
            copyfileobj(response, output)
        if digest(partial) != SHA256:
            raise ValueError("The downloaded football model failed its checksum.")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def digest(path: Path) -> str:
    checksum = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()
