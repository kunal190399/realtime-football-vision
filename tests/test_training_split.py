from pathlib import Path
import pytest
from src.training.football import prepare


def test_source_prefixes_do_not_leak_across_splits(tmp_path):
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "data.yaml").write_text("names: [ball, goalkeeper, player, referee]")
    for folder in ["train", "valid"]:
        for kind in ["images", "labels"]:
            (dataset / folder / kind).mkdir(parents=True)
        for group in range(10):
            name = f"clip{group}_{folder}"
            (dataset/folder/"images"/(name+".jpg")).write_bytes(b"\xff\xd8")
            (dataset/folder/"labels"/(name+".txt")).touch()
    output = tmp_path / "output"
    prepare(dataset, output)
    groups = []
    for split in ("train", "val", "test"):
        paths = (output/f"{split}.txt").read_text().splitlines()
        groups.append({Path(p).stem.split("_")[0] for p in paths})
    assert sum(map(len, groups)) == len(set.union(*groups)) == 10
    assert all(groups)
    with pytest.raises(ValueError):
        prepare(dataset, output)
