from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from soup_connectome.config import Scope
from soup_connectome.errors import (
    ArtifactExistsError,
    ChecksumMismatchError,
    GraphFormatError,
    GraphValidationError,
    PathContainmentError,
)

INT16_MIN = -(2**15)
INT16_MAX = 2**15 - 1
UINT16_MAX = 2**16 - 1
UINT32_MAX = 2**32 - 1
UINT64_MAX = 2**64 - 1
BLOCK_MAGIC = b"SCB1"
BLOCK_VERSION = 1
_BLOCK_HEADER = struct.Struct("<4sIIII")
_NEURON_RECORD = struct.Struct("<IQHbb")


@dataclass(frozen=True, slots=True)
class Edge:
    target: int
    weight: int
    delay: int


@dataclass(frozen=True, slots=True)
class GraphBlock:
    source_start: int
    rows: tuple[tuple[Edge, ...], ...]

    @property
    def source_count(self) -> int:
        return len(self.rows)

    @property
    def edge_count(self) -> int:
        return sum(len(row) for row in self.rows)


@dataclass(frozen=True, slots=True)
class NeuronRecord:
    external_id: int
    type_code: int = 0
    transmitter_code: int = 0
    side_code: int = 0


def _validate_block_edges(block: GraphBlock, n_neurons: int) -> None:
    if block.source_start < 0 or block.source_start + block.source_count > n_neurons:
        raise GraphValidationError("block source range is outside the graph")
    for row in block.rows:
        for edge in row:
            if not 0 <= edge.target < n_neurons:
                raise GraphValidationError("edge target is outside the graph")
            if not INT16_MIN <= edge.weight <= INT16_MAX:
                raise GraphValidationError("edge weight is outside int16 range")
            if not 1 <= edge.delay <= UINT16_MAX:
                raise GraphValidationError("edge delay must be within positive uint16 range")


@dataclass(frozen=True, slots=True)
class ConnectomeGraph:
    n_neurons: int
    neurons: tuple[NeuronRecord, ...]
    blocks: tuple[GraphBlock, ...]

    def __post_init__(self) -> None:
        self.validate()

    @property
    def edge_count(self) -> int:
        return sum(block.edge_count for block in self.blocks)

    @property
    def min_delay(self) -> int:
        delays = [edge.delay for block in self.blocks for row in block.rows for edge in row]
        return min(delays) if delays else 0

    @property
    def max_delay(self) -> int:
        delays = [edge.delay for block in self.blocks for row in block.rows for edge in row]
        return max(delays) if delays else 0

    def validate(self) -> None:
        if self.n_neurons < 0 or self.n_neurons > UINT32_MAX:
            raise GraphValidationError("neuron count is outside uint32 range")
        if len(self.neurons) != self.n_neurons:
            raise GraphValidationError("neuron table length does not match neuron count")
        for neuron in self.neurons:
            if not 0 <= neuron.external_id <= UINT64_MAX:
                raise GraphValidationError("external neuron id is outside uint64 range")
            if not 0 <= neuron.type_code <= 2**16 - 1:
                raise GraphValidationError("neuron type code is outside uint16 range")
            if not -(2**7) <= neuron.transmitter_code <= 2**7 - 1:
                raise GraphValidationError("neurotransmitter code is outside int8 range")
            if not -(2**7) <= neuron.side_code <= 2**7 - 1:
                raise GraphValidationError("side code is outside int8 range")

        expected_source = 0
        for block in self.blocks:
            if block.source_start != expected_source:
                raise GraphValidationError("blocks must cover contiguous source ranges")
            if block.source_start < 0 or block.source_count > UINT32_MAX:
                raise GraphValidationError("block source range is outside uint32 range")
            expected_source += block.source_count
            if expected_source > self.n_neurons:
                raise GraphValidationError("blocks contain more source neurons than the graph")
            for row in block.rows:
                for edge in row:
                    if not 0 <= edge.target < self.n_neurons:
                        raise GraphValidationError("edge target is outside the graph")
                    if not INT16_MIN <= edge.weight <= INT16_MAX:
                        raise GraphValidationError("edge weight is outside int16 range")
                    if not 1 <= edge.delay <= UINT16_MAX:
                        raise GraphValidationError(
                            "edge delay must be within positive uint16 range"
                        )
        if expected_source != self.n_neurons:
            raise GraphValidationError("blocks do not cover every source neuron")

    def iter_blocks(self) -> Iterator[GraphBlock]:
        """Yield resident blocks through the common graph-source interface."""

        yield from self.blocks


class BlockDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    source_start: int = Field(ge=0, le=UINT32_MAX)
    source_count: int = Field(ge=0, le=UINT32_MAX)
    edge_count: int = Field(ge=0, le=UINT32_MAX)
    byte_size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ConnectomeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format_id: str = "scx"
    format_version: int = 1
    dataset_id: str
    source_url: str | None = None
    license: str | None = None
    n_neurons: int = Field(ge=0, le=UINT32_MAX)
    n_edges: int = Field(ge=0, le=UINT32_MAX)
    voltage_scale_pow: int = 16
    index_dtype: str = "uint32"
    weight_dtype: str = "int16"
    delay_dtype: str = "uint16"
    endianness: str = "little"
    scope: Scope = Scope.full
    neurons_byte_size: int = Field(ge=0)
    neurons_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    conversion: dict[str, Any] = Field(default_factory=dict)
    blocks: tuple[BlockDescriptor, ...]


@dataclass(frozen=True, slots=True)
class GraphArtifact:
    path: Path
    manifest: ConnectomeManifest
    graph: ConnectomeGraph


@dataclass(slots=True)
class DiskGraphArtifact:
    """Manifest and neuron metadata with lazy, one-block-at-a-time loading."""

    path: Path
    manifest: ConnectomeManifest
    neurons: tuple[NeuronRecord, ...]
    verify_checksums: bool = True
    _max_delay: int | None = None

    @property
    def n_neurons(self) -> int:
        return self.manifest.n_neurons

    @property
    def edge_count(self) -> int:
        return self.manifest.n_edges

    @property
    def max_delay(self) -> int:
        if self._max_delay is None:
            self._max_delay = max(
                (edge.delay for block in self.iter_blocks() for row in block.rows for edge in row),
                default=0,
            )
        return self._max_delay

    def iter_blocks(self) -> Iterator[GraphBlock]:
        """Read, validate, and yield one block without retaining prior blocks."""

        expected_source = 0
        for descriptor in self.manifest.blocks:
            block_path = safe_join(self.path, descriptor.filename)
            try:
                data = block_path.read_bytes()
            except OSError as exc:
                raise GraphFormatError(f"cannot read block: {block_path}") from exc
            if self.verify_checksums:
                data = _verify_file(block_path, descriptor.byte_size, descriptor.sha256)
            block = deserialize_block(data)
            if (
                block.source_start != descriptor.source_start
                or block.source_count != descriptor.source_count
                or block.edge_count != descriptor.edge_count
            ):
                raise GraphFormatError(f"block descriptor does not match file: {block_path}")
            if block.source_start != expected_source:
                raise GraphValidationError("blocks must cover contiguous source ranges")
            _validate_block_edges(block, self.n_neurons)
            expected_source += block.source_count
            yield block
        if expected_source != self.n_neurons:
            raise GraphValidationError("blocks do not cover every source neuron")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_join(root: Path, relative: str | os.PathLike[str]) -> Path:
    """Join a path only when its real path remains inside ``root``."""

    root_real = os.path.realpath(os.fspath(root))
    candidate_real = os.path.realpath(os.path.join(root_real, os.fspath(relative)))
    try:
        contained = os.path.commonpath((root_real, candidate_real)) == root_real
    except ValueError as exc:
        raise PathContainmentError("artifact path uses a different drive") from exc
    if not contained:
        raise PathContainmentError(f"path escapes artifact root: {relative}")
    return Path(candidate_real)


def serialize_block(block: GraphBlock) -> bytes:
    if block.source_start < 0 or block.source_start > UINT32_MAX:
        raise GraphValidationError("block source start is outside uint32 range")
    if block.source_count > UINT32_MAX or block.edge_count > UINT32_MAX:
        raise GraphValidationError("block dimensions are outside uint32 range")

    offsets = [0]
    targets: list[int] = []
    weights: list[int] = []
    delays: list[int] = []
    for row in block.rows:
        for edge in row:
            if not 0 <= edge.target <= UINT32_MAX:
                raise GraphValidationError("edge target is outside uint32 range")
            if not INT16_MIN <= edge.weight <= INT16_MAX:
                raise GraphValidationError("edge weight is outside int16 range")
            if not 1 <= edge.delay <= UINT16_MAX:
                raise GraphValidationError("edge delay must be within positive uint16 range")
            targets.append(edge.target)
            weights.append(edge.weight)
            delays.append(edge.delay)
        offsets.append(len(targets))

    encoded = bytearray(
        _BLOCK_HEADER.pack(
            BLOCK_MAGIC,
            BLOCK_VERSION,
            block.source_start,
            block.source_count,
            len(targets),
        )
    )
    encoded.extend(struct.pack(f"<{len(offsets)}I", *offsets))
    if targets:
        encoded.extend(struct.pack(f"<{len(targets)}I", *targets))
        encoded.extend(struct.pack(f"<{len(weights)}h", *weights))
        encoded.extend(struct.pack(f"<{len(delays)}H", *delays))
    return bytes(encoded)


def deserialize_block(data: bytes) -> GraphBlock:
    if len(data) < _BLOCK_HEADER.size:
        raise GraphFormatError("block is shorter than its header")
    magic, version, source_start, source_count, edge_count = _BLOCK_HEADER.unpack_from(data)
    if magic != BLOCK_MAGIC:
        raise GraphFormatError("block magic is invalid")
    if version != BLOCK_VERSION:
        raise GraphFormatError(f"unsupported block version: {version}")

    offsets_count = source_count + 1
    expected_size = (
        _BLOCK_HEADER.size + offsets_count * 4 + edge_count * 4 + edge_count * 2 + edge_count * 2
    )
    if len(data) != expected_size:
        raise GraphFormatError("block byte size does not match its header")

    offset = _BLOCK_HEADER.size
    offsets = struct.unpack_from(f"<{offsets_count}I", data, offset)
    offset += offsets_count * 4
    if (
        offsets[0] != 0
        or offsets[-1] != edge_count
        or any(a > b for a, b in zip(offsets, offsets[1:]))
    ):
        raise GraphFormatError("CSR offsets are not monotonic")
    targets = struct.unpack_from(f"<{edge_count}I", data, offset) if edge_count else ()
    offset += edge_count * 4
    weights = struct.unpack_from(f"<{edge_count}h", data, offset) if edge_count else ()
    offset += edge_count * 2
    delays = struct.unpack_from(f"<{edge_count}H", data, offset) if edge_count else ()
    rows: list[tuple[Edge, ...]] = []
    for start, end in zip(offsets, offsets[1:]):
        rows.append(tuple(Edge(targets[i], weights[i], delays[i]) for i in range(start, end)))
    try:
        return GraphBlock(source_start=source_start, rows=tuple(rows))
    except (IndexError, ValueError) as exc:
        raise GraphFormatError("block rows are malformed") from exc


def _serialize_neurons(neurons: tuple[NeuronRecord, ...]) -> bytes:
    encoded = bytearray()
    for index, neuron in enumerate(neurons):
        encoded.extend(_serialize_neuron(index, neuron))
    return bytes(encoded)


def _serialize_neuron(index: int, neuron: NeuronRecord) -> bytes:
    try:
        return _NEURON_RECORD.pack(
            index,
            neuron.external_id,
            neuron.type_code,
            neuron.transmitter_code,
            neuron.side_code,
        )
    except struct.error as exc:
        raise GraphValidationError("neuron record is outside its binary dtype range") from exc


def _write_neurons(
    path: Path,
    neurons: Iterable[NeuronRecord],
    expected_count: int,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_size = 0
    actual_count = 0
    with path.open("wb") as stream:
        for index, neuron in enumerate(neurons):
            encoded = _serialize_neuron(index, neuron)
            stream.write(encoded)
            digest.update(encoded)
            byte_size += len(encoded)
            actual_count = index + 1
    if actual_count != expected_count:
        raise GraphValidationError(
            "neuron iterable length does not match neuron count: "
            f"{actual_count} != {expected_count}"
        )
    return byte_size, digest.hexdigest()


def _deserialize_neurons(data: bytes, expected_count: int) -> tuple[NeuronRecord, ...]:
    if len(data) != expected_count * _NEURON_RECORD.size:
        raise GraphFormatError("neuron table byte size does not match manifest")
    neurons = []
    for offset in range(0, len(data), _NEURON_RECORD.size):
        index, external_id, type_code, transmitter_code, side_code = _NEURON_RECORD.unpack_from(
            data, offset
        )
        if index != len(neurons):
            raise GraphFormatError("neuron dense indices are not contiguous")
        neurons.append(NeuronRecord(external_id, type_code, transmitter_code, side_code))
    return tuple(neurons)


def write_artifact(
    graph: ConnectomeGraph,
    destination: Path,
    *,
    dataset_id: str = "example",
    source_url: str | None = None,
    license: str | None = None,
    scope: Scope = Scope.full,
    conversion_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a new deterministic artifact and refuse to overwrite an existing one."""

    graph.validate()
    return write_artifact_from_blocks(
        graph.n_neurons,
        graph.neurons,
        graph.blocks,
        destination,
        dataset_id=dataset_id,
        source_url=source_url,
        license=license,
        scope=scope,
        conversion_metadata=conversion_metadata,
    )


def write_artifact_from_blocks(
    n_neurons: int,
    neurons: Iterable[NeuronRecord],
    blocks: Iterable[GraphBlock],
    destination: Path,
    *,
    dataset_id: str = "local",
    source_url: str | None = None,
    license: str | None = None,
    scope: Scope = Scope.full,
    conversion_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write blocks incrementally so a caller need not hold the whole graph."""

    if n_neurons < 0 or n_neurons > UINT32_MAX:
        raise GraphValidationError("neuron count is outside uint32 range")
    destination = Path(destination)
    if destination.exists():
        raise ArtifactExistsError(f"artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    blocks_dir = safe_join(destination, "blocks")
    blocks_dir.mkdir(parents=True, exist_ok=False)

    neurons_path = safe_join(destination, "neurons.bin")
    neurons_byte_size, neurons_sha256 = _write_neurons(neurons_path, neurons, n_neurons)
    descriptors = []
    expected_source = 0
    edge_count = 0
    for block_index, block in enumerate(blocks):
        if block.source_start != expected_source:
            raise GraphValidationError("blocks must cover contiguous source ranges")
        filename = f"block_{block_index:05d}.scb"
        data = serialize_block(block)
        block_path = safe_join(blocks_dir, filename)
        block_path.write_bytes(data)
        expected_source += block.source_count
        edge_count += block.edge_count
        descriptors.append(
            BlockDescriptor(
                filename=f"blocks/{filename}",
                source_start=block.source_start,
                source_count=block.source_count,
                edge_count=block.edge_count,
                byte_size=len(data),
                sha256=_sha256(data),
            )
        )
    if expected_source != n_neurons:
        raise GraphValidationError("blocks do not cover every source neuron")

    manifest = ConnectomeManifest(
        dataset_id=dataset_id,
        source_url=source_url,
        license=license,
        n_neurons=n_neurons,
        n_edges=edge_count,
        neurons_byte_size=neurons_byte_size,
        neurons_sha256=neurons_sha256,
        scope=scope,
        conversion=conversion_metadata or {},
        blocks=tuple(descriptors),
    )
    safe_join(destination, "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _verify_file(path: Path, expected_size: int, expected_sha256: str) -> bytes:
    data = path.read_bytes()
    if len(data) != expected_size:
        raise ChecksumMismatchError(f"byte size mismatch: {path}")
    if _sha256(data) != expected_sha256:
        raise ChecksumMismatchError(f"checksum mismatch: {path}")
    return data


def _read_manifest(destination: Path) -> ConnectomeManifest:
    manifest_path = safe_join(destination, "manifest.json")
    try:
        manifest = ConnectomeManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GraphFormatError(f"cannot read manifest: {manifest_path}") from exc
    if manifest.format_id != "scx" or manifest.format_version != 1:
        raise GraphFormatError("unsupported connectome artifact format")
    return manifest


def _read_neurons(
    destination: Path,
    manifest: ConnectomeManifest,
    *,
    verify_checksums: bool,
) -> tuple[NeuronRecord, ...]:
    neurons_path = safe_join(destination, "neurons.bin")
    try:
        neurons_data = neurons_path.read_bytes()
    except OSError as exc:
        raise GraphFormatError(f"cannot read neuron table: {neurons_path}") from exc
    if verify_checksums:
        neurons_data = _verify_file(
            neurons_path, manifest.neurons_byte_size, manifest.neurons_sha256
        )
    return _deserialize_neurons(neurons_data, manifest.n_neurons)


def open_artifact(destination: Path, *, verify_checksums: bool = True) -> DiskGraphArtifact:
    """Open an artifact without materializing its connection blocks."""

    destination = Path(destination)
    manifest = _read_manifest(destination)
    neurons = _read_neurons(destination, manifest, verify_checksums=verify_checksums)
    return DiskGraphArtifact(destination, manifest, neurons, verify_checksums)


def load_artifact(destination: Path, *, verify_checksums: bool = True) -> GraphArtifact:
    """Load an artifact and materialize all blocks in memory."""

    streamed = open_artifact(destination, verify_checksums=verify_checksums)
    blocks = tuple(streamed.iter_blocks())
    try:
        graph = ConnectomeGraph(streamed.n_neurons, streamed.neurons, blocks)
    except GraphValidationError:
        raise
    except Exception as exc:
        raise GraphFormatError("loaded graph failed validation") from exc
    if graph.edge_count != streamed.edge_count:
        raise GraphFormatError("manifest edge count does not match blocks")
    return GraphArtifact(streamed.path, streamed.manifest, graph)
