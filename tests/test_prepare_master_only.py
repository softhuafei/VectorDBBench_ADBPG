from __future__ import annotations

from pathlib import Path  # noqa: TC003

import pytest

from scripts.prepare_hybrid_master_only import discover_shards


def test_discover_shards_is_sorted_and_can_be_limited(tmp_path: Path):
    for name in [
        "shuffle_train-02-of-03.parquet",
        "shuffle_train-00-of-03.parquet",
        "shuffle_train-01-of-03.parquet",
    ]:
        (tmp_path / name).touch()

    shards = discover_shards(tmp_path, "shuffle_train-*-of-*.parquet", max_shards=2)

    assert [path.name for path in shards] == [
        "shuffle_train-00-of-03.parquet",
        "shuffle_train-01-of-03.parquet",
    ]


def test_discover_shards_ignores_directories(tmp_path: Path):
    (tmp_path / "shuffle_train-00-of-01.parquet").mkdir()
    with pytest.raises(FileNotFoundError, match="no Parquet shards"):
        discover_shards(tmp_path, "shuffle_train-*-of-*.parquet", max_shards=None)
