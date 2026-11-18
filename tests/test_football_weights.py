from hashlib import sha256
from io import BytesIO

import pytest

from src.detection import football_weights as weights


def test_download_replaces_corrupt_cache_and_reuses_verified_file(tmp_path, monkeypatch):
    contents = b"model data"
    requests = []

    def download(url, timeout):
        requests.append(url)
        return BytesIO(contents)

    monkeypatch.setattr(weights, "SHA256", sha256(contents).hexdigest())
    monkeypatch.setattr(weights, "urlopen", download)
    path = weights.football_weights(tmp_path)
    assert weights.football_weights(tmp_path) == path
    assert len(requests) == 1
    path.write_bytes(b"corrupt")
    assert weights.football_weights(tmp_path).read_bytes() == contents
    assert len(requests) == 2


def test_download_checksum_failure_leaves_no_partial_model(tmp_path, monkeypatch):
    monkeypatch.setattr(weights, "urlopen", lambda *args, **kwargs: BytesIO(b"bad data"))
    with pytest.raises(ValueError, match="checksum"):
        weights.football_weights(tmp_path)
    assert list(tmp_path.iterdir()) == []
