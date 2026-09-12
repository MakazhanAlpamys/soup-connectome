from pathlib import Path

import pytest

from soup_connectome.errors import (
    ArtifactExistsError,
    ChecksumMismatchError,
    GraphFormatError,
    PathContainmentError,
)
from soup_connectome.graph.builder import build_artifact_from_rows
from soup_connectome.graph.example import example_graph
from soup_connectome.graph.format import (
    Edge,
    load_artifact,
    safe_join,
    serialize_block,
    write_artifact,
)


def test_example_artifact_round_trips(tmp_path: Path) -> None:
    graph = example_graph()
    artifact_path = write_artifact(graph, tmp_path / "example.scx")

    loaded = load_artifact(artifact_path, verify_checksums=True)

    assert loaded.graph == graph
    assert loaded.manifest.n_neurons == graph.n_neurons
    assert loaded.manifest.n_edges == graph.edge_count


def test_row_builder_emits_incremental_source_blocks(tmp_path: Path) -> None:
    artifact_path = build_artifact_from_rows(
        3,
        ((0, Edge(target=1, weight=7, delay=1)), (2, Edge(target=0, weight=-2, delay=2))),
        tmp_path / "rows.scx",
        block_size=2,
    )

    loaded = load_artifact(artifact_path)

    assert [block.source_count for block in loaded.graph.blocks] == [2, 1]
    assert loaded.graph.edge_count == 2


def test_writer_refuses_to_overwrite_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "example.scx"
    write_artifact(example_graph(), destination)

    with pytest.raises(ArtifactExistsError):
        write_artifact(example_graph(), destination)


def test_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    artifact_path = write_artifact(example_graph(), tmp_path / "example.scx")
    block_path = next((artifact_path / "blocks").glob("*.scb"))
    block_bytes = bytearray(block_path.read_bytes())
    block_bytes[-1] ^= 1
    block_path.write_bytes(block_bytes)

    with pytest.raises(ChecksumMismatchError):
        load_artifact(artifact_path, verify_checksums=True)


def test_malformed_block_header_is_rejected() -> None:
    block = example_graph().blocks[0]
    encoded = bytearray(serialize_block(block))
    encoded[0:4] = b"BAD!"

    with pytest.raises(GraphFormatError, match="magic"):
        from soup_connectome.graph.format import deserialize_block

        deserialize_block(bytes(encoded))


def test_path_containment_uses_real_paths(tmp_path: Path) -> None:
    with pytest.raises(PathContainmentError):
        safe_join(tmp_path, Path("..") / "outside.scb")
