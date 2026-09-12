from __future__ import annotations

import hashlib
import heapq
import json
import struct
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from soup_connectome.config import Scope
from soup_connectome.errors import ArtifactExistsError, DataSchemaError, QuantizationError
from soup_connectome.graph.builder import build_artifact_from_rows
from soup_connectome.graph.format import Edge, NeuronRecord


class MaleCNSColumns(BaseModel):
    """Column mapping for the three local MaleCNS Feather tables.

    The connection column defaults are the verified names from the handoff.
    Annotation and neurotransmitter names are intentionally explicit because
    this adapter must not guess source schema or semantic fields.
    """

    model_config = ConfigDict(extra="forbid")

    weight_pre: str = "body_pre"
    weight_post: str = "body_post"
    weight_value: str = "weight"
    annotation_id: str
    annotation_type: str | None = None
    annotation_side: str | None = None
    neurotransmitter_id: str
    neurotransmitter_name: str


class WeightQuantizer(BaseModel):
    """Deterministic rational quantizer for signed int16 edge impulses."""

    model_config = ConfigDict(extra="forbid")

    numerator: int = Field(default=1, gt=0)
    denominator: int = Field(default=1, gt=0)
    rounding: Literal["nearest_even"] = "nearest_even"
    overflow: Literal["reject", "saturate"] = "reject"

    def quantize(self, value: int) -> int:
        result, _ = self.quantize_with_status(value)
        return result

    def quantize_with_status(self, value: int) -> tuple[int, bool]:
        scaled = _round_nearest_even(value * self.numerator, self.denominator)
        if -(2**15) <= scaled <= 2**15 - 1:
            return scaled, False
        if self.overflow == "reject":
            raise QuantizationError(f"value cannot fit signed int16: {scaled}")
        return max(-(2**15), min(2**15 - 1, scaled)), True


@dataclass(frozen=True, slots=True)
class ConversionReport:
    artifact_path: Path
    n_neurons: int
    n_edges: int
    excluded_edges: int
    saturated_weights: int


_EDGE_RUN = struct.Struct("<qqi")
_DEFAULT_BATCH_SIZE = 65536
_DEFAULT_SORT_CHUNK_SIZE = 100000


def _round_nearest_even(numerator: int, denominator: int) -> int:
    sign = -1 if numerator < 0 else 1
    absolute = abs(numerator)
    quotient, remainder = divmod(absolute, denominator)
    twice_remainder = remainder * 2
    if twice_remainder > denominator or (twice_remainder == denominator and quotient % 2 == 1):
        quotient += 1
    return sign * quotient


def _require_local_file(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise DataSchemaError(f"{label} must be an existing local Feather file: {path}")
    return path


def iter_feather_rows(
    path: Path,
    columns: Sequence[str],
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> Iterator[tuple[Any, ...]]:
    """Yield selected Feather columns in bounded scanner batches.

    The default batch size is a chosen I/O chunk estimate, not a measured
    memory or throughput bound. PyArrow is imported only when this adapter runs.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    path = _require_local_file(path, "source")
    try:
        import pyarrow.dataset as dataset_api

        dataset = dataset_api.dataset(path, format=dataset_api.IpcFileFormat())
        available = set(dataset.schema.names)
    except ImportError as exc:
        raise DataSchemaError("MaleCNS conversion requires optional dependency pyarrow") from exc
    except Exception as exc:
        raise DataSchemaError(f"cannot open Feather source: {path}") from exc

    missing = [column for column in columns if column not in available]
    if missing:
        raise DataSchemaError(f"missing columns in {path.name}: {', '.join(missing)}")
    try:
        scanner = dataset.scanner(
            columns=list(columns),
            batch_size=batch_size,
            use_threads=False,
        )
        for batch in scanner.to_batches():
            arrays = [batch.column(index) for index in range(len(columns))]
            for row_index in range(batch.num_rows):
                yield tuple(array[row_index].as_py() for array in arrays)
    except DataSchemaError:
        raise
    except Exception as exc:
        raise DataSchemaError(f"cannot scan Feather source: {path}") from exc


def feather_columns(path: Path) -> tuple[str, ...]:
    """Return a local Feather schema without loading table rows."""

    path = _require_local_file(path, "source")
    try:
        import pyarrow.dataset as dataset_api

        dataset = dataset_api.dataset(path, format=dataset_api.IpcFileFormat())
    except ImportError as exc:
        raise DataSchemaError("MaleCNS inspection requires optional dependency pyarrow") from exc
    except Exception as exc:
        raise DataSchemaError(f"cannot open Feather source: {path}") from exc
    return tuple(dataset.schema.names)


def _as_body_id(value: Any, field: str) -> int:
    if isinstance(value, bool) or value is None:
        raise DataSchemaError(f"{field} contains a non-integer body id")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise DataSchemaError(f"{field} contains a non-integer body id: {value!r}") from exc
    if converted != value or converted < 0 or converted > 2**64 - 1:
        raise DataSchemaError(f"{field} body id is outside uint64 range: {value!r}")
    return converted


def _as_raw_weight(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        raise DataSchemaError("weight contains a non-integer value")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise DataSchemaError(f"weight contains a non-integer value: {value!r}") from exc
    if converted != value or not -(2**31) <= converted <= 2**31 - 1:
        raise DataSchemaError(f"weight is outside signed int32 range: {value!r}")
    return converted


def _category_label(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value)


def _category_codes(labels: Iterable[str], maximum: int) -> dict[str, int]:
    unique = sorted(set(labels) | {"unknown"})
    if len(unique) > maximum + 1:
        raise DataSchemaError("annotation category count exceeds target integer range")
    return {label: index for index, label in enumerate(unique)}


def _load_annotations(
    path: Path,
    columns: MaleCNSColumns,
    *,
    batch_size: int,
) -> dict[int, tuple[str, str]]:
    selected = [columns.annotation_id]
    if columns.annotation_type:
        selected.append(columns.annotation_type)
    if columns.annotation_side:
        selected.append(columns.annotation_side)
    annotations: dict[int, tuple[str, str]] = {}
    for row in iter_feather_rows(path, selected, batch_size=batch_size):
        body_id = _as_body_id(row[0], columns.annotation_id)
        type_label = _category_label(row[1]) if columns.annotation_type else "unknown"
        side_offset = 2 if columns.annotation_type else 1
        side_label = _category_label(row[side_offset]) if columns.annotation_side else "unknown"
        record = (type_label, side_label)
        if body_id in annotations and annotations[body_id] != record:
            raise DataSchemaError(f"conflicting annotation rows for body id: {body_id}")
        annotations[body_id] = record
    return annotations


def _load_neurotransmitters(
    path: Path,
    columns: MaleCNSColumns,
    *,
    batch_size: int,
) -> dict[int, str]:
    selected = [columns.neurotransmitter_id, columns.neurotransmitter_name]
    neurotransmitters: dict[int, str] = {}
    for row in iter_feather_rows(path, selected, batch_size=batch_size):
        body_id = _as_body_id(row[0], columns.neurotransmitter_id)
        label = _category_label(row[1])
        if body_id in neurotransmitters and neurotransmitters[body_id] != label:
            raise DataSchemaError(f"conflicting neurotransmitter rows for body id: {body_id}")
        neurotransmitters[body_id] = label
    return neurotransmitters


def _weight_rows(
    path: Path,
    columns: MaleCNSColumns,
    *,
    batch_size: int,
) -> Iterator[tuple[int, int, int]]:
    selected = [columns.weight_pre, columns.weight_post, columns.weight_value]
    for row in iter_feather_rows(path, selected, batch_size=batch_size):
        yield (
            _as_body_id(row[0], columns.weight_pre),
            _as_body_id(row[1], columns.weight_post),
            _as_raw_weight(row[2]),
        )


def _feather_endpoint_ids(
    path: Path,
    columns: MaleCNSColumns,
    *,
    batch_size: int,
) -> set[int]:
    """Collect unique endpoints with Arrow kernels instead of Python row loops."""

    path = _require_local_file(path, "weights")
    try:
        import pyarrow.compute as compute_api
        import pyarrow.dataset as dataset_api

        dataset = dataset_api.dataset(path, format=dataset_api.IpcFileFormat())
        available = set(dataset.schema.names)
    except ImportError as exc:
        raise DataSchemaError("MaleCNS conversion requires optional dependency pyarrow") from exc
    except Exception as exc:
        raise DataSchemaError(f"cannot open Feather source: {path}") from exc

    selected = [columns.weight_pre, columns.weight_post]
    missing = [column for column in selected if column not in available]
    if missing:
        raise DataSchemaError(f"missing columns in {path.name}: {', '.join(missing)}")
    endpoint_ids: set[int] = set()
    try:
        scanner = dataset.scanner(columns=selected, batch_size=batch_size, use_threads=False)
        for batch in scanner.to_batches():
            for index, column in enumerate(selected):
                for value in compute_api.unique(batch.column(index)).to_pylist():
                    endpoint_ids.add(_as_body_id(value, column))
    except DataSchemaError:
        raise
    except Exception as exc:
        raise DataSchemaError(f"cannot scan Feather source: {path}") from exc
    return endpoint_ids


def _write_edge_run(path: Path, edges: list[tuple[int, int, int]]) -> None:
    edges.sort()
    with path.open("wb") as stream:
        for edge in edges:
            stream.write(_EDGE_RUN.pack(*edge))


def _read_edge_run(path: Path) -> Iterator[tuple[int, int, int]]:
    with path.open("rb") as stream:
        while data := stream.read(_EDGE_RUN.size):
            if len(data) != _EDGE_RUN.size:
                raise DataSchemaError(f"truncated temporary sort run: {path}")
            yield _EDGE_RUN.unpack(data)


def _sorted_weight_rows(
    path: Path,
    columns: MaleCNSColumns,
    *,
    batch_size: int,
    sort_chunk_size: int,
    temporary_directory: Path,
) -> Iterator[tuple[int, int, int]]:
    if sort_chunk_size <= 0:
        raise ValueError("sort_chunk_size must be positive")
    runs: list[Path] = []
    chunk: list[tuple[int, int, int]] = []
    for row in _weight_rows(path, columns, batch_size=batch_size):
        chunk.append(row)
        if len(chunk) >= sort_chunk_size:
            run_path = temporary_directory / f"run_{len(runs):05d}.bin"
            _write_edge_run(run_path, chunk)
            runs.append(run_path)
            chunk = []
    if chunk:
        run_path = temporary_directory / f"run_{len(runs):05d}.bin"
        _write_edge_run(run_path, chunk)
        runs.append(run_path)
    yield from heapq.merge(*(_read_edge_run(run) for run in runs))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def convert_male_cns(
    weights_path: Path,
    annotations_path: Path,
    neurotransmitters_path: Path,
    destination: Path,
    *,
    columns: MaleCNSColumns,
    sign_mapping: dict[str, int],
    block_size: int,
    excluded_neurotransmitters: Sequence[str] = (),
    quantizer: WeightQuantizer | None = None,
    scope: Scope = Scope.full,
    delay_steps: int = 1,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    sort_chunk_size: int = _DEFAULT_SORT_CHUNK_SIZE,
) -> ConversionReport:
    """Convert three local MaleCNS tables into a deterministic `.scx` artifact.

    ``delay_steps=1`` is a representation fallback required by the current
    graph format, not a measured MaleCNS axonal delay. A future biological
    calibration must replace it explicitly.
    """

    weights_path = _require_local_file(weights_path, "weights")
    annotations_path = _require_local_file(annotations_path, "annotations")
    neurotransmitters_path = _require_local_file(neurotransmitters_path, "neurotransmitters")
    destination = Path(destination)
    if destination.exists():
        raise ArtifactExistsError(f"artifact already exists: {destination}")
    scope = Scope(scope)
    if delay_steps < 1 or delay_steps > 2**16 - 1:
        raise DataSchemaError("delay_steps must fit positive uint16")
    if not sign_mapping:
        raise DataSchemaError("sign mapping must not be empty")
    normalized_signs: dict[str, int] = {}
    for label, sign in sign_mapping.items():
        if sign not in (-1, 1):
            raise DataSchemaError(f"sign mapping must use -1 or 1: {label}={sign}")
        normalized_signs[str(label)] = sign
    excluded_labels = {str(label) for label in excluded_neurotransmitters}
    overlap = sorted(excluded_labels & normalized_signs.keys())
    if overlap:
        raise DataSchemaError(
            "a neurotransmitter cannot be both signed and excluded: "
            + ", ".join(overlap)
        )
    quantizer = quantizer or WeightQuantizer()
    annotations = _load_annotations(annotations_path, columns, batch_size=batch_size)
    neurotransmitters = _load_neurotransmitters(
        neurotransmitters_path, columns, batch_size=batch_size
    )
    if scope is Scope.full:
        external_ids = set(annotations) | _feather_endpoint_ids(
            weights_path, columns, batch_size=batch_size
        )
    else:
        endpoint_ids: set[int] = set()
        for pre_external, post_external, _ in _weight_rows(
            weights_path, columns, batch_size=batch_size
        ):
            neurotransmitter = neurotransmitters.get(pre_external, "unknown")
            if neurotransmitter in excluded_labels:
                continue
            endpoint_ids.update((pre_external, post_external))
        external_ids = endpoint_ids
    sorted_external_ids = tuple(sorted(external_ids))
    dense_by_external = {
        external_id: index for index, external_id in enumerate(sorted_external_ids)
    }
    type_labels = [
        annotations.get(body_id, ("unknown", "unknown"))[0] for body_id in sorted_external_ids
    ]
    side_labels = [
        annotations.get(body_id, ("unknown", "unknown"))[1] for body_id in sorted_external_ids
    ]
    type_codes = _category_codes(type_labels, 2**16 - 1)
    side_codes = _category_codes(side_labels, 2**7 - 1)
    neurons = tuple(
        NeuronRecord(
            external_id=body_id,
            type_code=type_codes[annotations.get(body_id, ("unknown", "unknown"))[0]],
            transmitter_code=(normalized_signs.get(neurotransmitters.get(body_id, "unknown"), 0)),
            side_code=side_codes[annotations.get(body_id, ("unknown", "unknown"))[1]],
        )
        for body_id in sorted_external_ids
    )
    saturated_weights = 0
    edge_count = 0
    excluded_edges = 0

    def converted_rows(temporary_directory: Path) -> Iterator[tuple[int, Edge]]:
        nonlocal edge_count, excluded_edges, saturated_weights
        for pre_external, post_external, raw_weight in _sorted_weight_rows(
            weights_path,
            columns,
            batch_size=batch_size,
            sort_chunk_size=sort_chunk_size,
            temporary_directory=temporary_directory,
        ):
            neurotransmitter = neurotransmitters.get(pre_external, "unknown")
            if neurotransmitter in excluded_labels:
                excluded_edges += 1
                continue
            if neurotransmitter not in normalized_signs:
                raise DataSchemaError(
                    f"sign mapping has no entry for source neurotransmitter: {neurotransmitter}"
                )
            try:
                source_index = dense_by_external[pre_external]
                target_index = dense_by_external[post_external]
            except KeyError as exc:
                raise DataSchemaError(
                    "included edge endpoint is missing from the annotation table: "
                    f"{exc.args[0]}"
                ) from exc
            signed_weight = raw_weight * normalized_signs[neurotransmitter]
            quantized_weight, saturated = quantizer.quantize_with_status(signed_weight)
            edge_count += 1
            saturated_weights += int(saturated)
            yield (
                source_index,
                Edge(
                    target=target_index,
                    weight=quantized_weight,
                    delay=delay_steps,
                ),
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".soup-connectome-convert-", dir=destination.parent
    ) as temporary_root:
        temporary_root_path = Path(temporary_root)
        staging = temporary_root_path / "artifact.scx"
        conversion_metadata = {
            "adapter": "male-cns",
            "weights_sha256": _file_sha256(weights_path),
            "annotations_sha256": _file_sha256(annotations_path),
            "neurotransmitters_sha256": _file_sha256(neurotransmitters_path),
            "columns": columns.model_dump(mode="json"),
            "sign_mapping": normalized_signs,
            "excluded_neurotransmitters": sorted(excluded_labels),
            "quantizer": quantizer.model_dump(mode="json"),
            "delay_steps": delay_steps,
            "scope_rule": (
                "all annotation IDs plus raw endpoints for full; "
                "endpoints of included edges for compact"
            ),
            "type_category_codes": type_codes,
            "side_category_codes": side_codes,
            "excluded_edges": 0,
            "saturated_weights": 0,
        }
        build_artifact_from_rows(
            len(sorted_external_ids),
            converted_rows(temporary_root_path),
            staging,
            block_size=block_size,
            neurons=neurons,
            dataset_id="male-cns:v1.0",
            source_url="https://male-cns.janelia.org/download/",
            license="CC-BY",
            scope=scope,
            conversion_metadata=conversion_metadata,
        )
        conversion_metadata["saturated_weights"] = saturated_weights
        final_manifest = staging / "manifest.json"
        manifest_text = final_manifest.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
        manifest["conversion"]["excluded_edges"] = excluded_edges
        manifest["conversion"]["saturated_weights"] = saturated_weights
        final_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.replace(destination)
    return ConversionReport(
        artifact_path=destination,
        n_neurons=len(sorted_external_ids),
        n_edges=edge_count,
        excluded_edges=excluded_edges,
        saturated_weights=saturated_weights,
    )
