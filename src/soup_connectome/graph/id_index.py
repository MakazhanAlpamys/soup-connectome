from __future__ import annotations

import sqlite3
import struct
from collections.abc import Iterable, Iterator
from pathlib import Path

from soup_connectome.errors import DataSchemaError

_UINT64 = struct.Struct(">Q")
_INSERT_BATCH_SIZE = 4096  # chosen implementation batch size, not a measured memory bound


class DiskBackedIdIndex:
    """Map sorted uint64 source IDs to deterministic dense indices on disk.

    The SQLite table is temporary conversion state, not part of the `.scx`
    artifact. IDs are inserted in ascending order, so SQLite's rowid provides
    the dense index without materializing an ``external_id -> index`` dict.
    The index intentionally uses a big-endian blob key because SQLite INTEGER
    cannot represent the complete uint64 range.
    """

    def __init__(self, path: Path, connection: sqlite3.Connection) -> None:
        self.path = Path(path)
        self._connection: sqlite3.Connection | None = connection

    @classmethod
    def from_sorted_unique(
        cls,
        path: Path,
        external_ids: Iterable[int],
    ) -> DiskBackedIdIndex:
        """Build an index from non-decreasing uint64 IDs.

        Duplicate IDs are accepted and collapsed. The caller must provide the
        IDs in non-decreasing numeric order so the assigned dense indices are
        stable and independent of SQLite query-plan details.
        """

        path = Path(path)
        if path.exists():
            raise FileExistsError(f"index already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        try:
            connection.execute("CREATE TABLE nodes (external_id BLOB NOT NULL UNIQUE)")
            batch: list[tuple[bytes]] = []
            previous: int | None = None
            for external_id in external_ids:
                if isinstance(external_id, bool) or not isinstance(external_id, int):
                    raise ValueError("external IDs must be integers")
                if not 0 <= external_id <= 2**64 - 1:
                    raise ValueError("external ID is outside uint64 range")
                if previous is not None and external_id < previous:
                    raise ValueError("external IDs must be sorted")
                if external_id == previous:
                    continue
                batch.append((_UINT64.pack(external_id),))
                previous = external_id
                if len(batch) >= _INSERT_BATCH_SIZE:
                    connection.executemany("INSERT INTO nodes(external_id) VALUES (?)", batch)
                    batch.clear()
            if batch:
                connection.executemany("INSERT INTO nodes(external_id) VALUES (?)", batch)
            connection.commit()
        except Exception:
            connection.close()
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            raise
        return cls(path, connection)

    @property
    def count(self) -> int:
        connection = self._require_open()
        row = connection.execute("SELECT COUNT(*) FROM nodes").fetchone()
        return int(row[0])

    def lookup(self, external_id: int) -> int:
        """Return the dense index for one source ID, or raise ``KeyError``."""

        if isinstance(external_id, bool) or not isinstance(external_id, int):
            raise KeyError(external_id)
        if not 0 <= external_id <= 2**64 - 1:
            raise KeyError(external_id)
        connection = self._require_open()
        row = connection.execute(
            "SELECT rowid FROM nodes WHERE external_id = ?", (_UINT64.pack(external_id),)
        ).fetchone()
        if row is None:
            raise KeyError(external_id)
        return int(row[0]) - 1

    def iter_dense(self) -> Iterator[tuple[int, int]]:
        """Yield ``(dense_index, external_id)`` in canonical dense order."""

        connection = self._require_open()
        expected_dense = 0
        for rowid, packed_id in connection.execute(
            "SELECT rowid, external_id FROM nodes ORDER BY rowid"
        ):
            if rowid - 1 != expected_dense or len(packed_id) != _UINT64.size:
                raise DataSchemaError("disk-backed ID index dense rows are malformed")
            yield expected_dense, _UINT64.unpack(packed_id)[0]
            expected_dense += 1

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> DiskBackedIdIndex:
        self._require_open()
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        self.close()

    def _require_open(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("disk-backed ID index is closed")
        return self._connection
