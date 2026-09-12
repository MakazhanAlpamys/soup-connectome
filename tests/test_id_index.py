from pathlib import Path

import pytest

from soup_connectome.graph.id_index import DiskBackedIdIndex


def test_disk_backed_id_index_is_sorted_and_uint64_safe(tmp_path: Path) -> None:
    index = DiskBackedIdIndex.from_sorted_unique(
        tmp_path / "nodes.sqlite3",
        (0, 2**32, 2**63, 2**64 - 1),
    )

    try:
        assert index.count == 4
        assert index.lookup(0) == 0
        assert index.lookup(2**32) == 1
        assert index.lookup(2**63) == 2
        assert index.lookup(2**64 - 1) == 3
        assert list(index.iter_dense()) == [
            (0, 0),
            (1, 2**32),
            (2, 2**63),
            (3, 2**64 - 1),
        ]
    finally:
        index.close()


def test_disk_backed_id_index_rejects_unsorted_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sorted"):
        DiskBackedIdIndex.from_sorted_unique(tmp_path / "nodes.sqlite3", (2, 1))


def test_disk_backed_id_index_rejects_missing_id(tmp_path: Path) -> None:
    index = DiskBackedIdIndex.from_sorted_unique(tmp_path / "nodes.sqlite3", (1,))
    try:
        with pytest.raises(KeyError):
            index.lookup(2)
    finally:
        index.close()
