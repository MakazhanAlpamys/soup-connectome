from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_connectome.cli import app
from soup_connectome.errors import DataSchemaError, QuantizationError
from soup_connectome.graph.format import load_artifact
from soup_connectome.graph.malecns import (
    MaleCNSColumns,
    WeightQuantizer,
    convert_male_cns,
    feather_columns,
)


def _write_feather(path: Path, columns: dict[str, list[object]]) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.feather as feather

    feather.write_feather(pyarrow.table(columns), path)


@pytest.fixture
def malecns_tables(tmp_path: Path) -> tuple[Path, Path, Path]:
    weights = tmp_path / "weights.feather"
    annotations = tmp_path / "annotations.feather"
    neurotransmitters = tmp_path / "neurotransmitters.feather"
    _write_feather(
        weights,
        {
            "pre_id": [200, 100, 100],
            "post_id": [300, 300, 200],
            "strength": [4, 2, 3],
        },
    )
    _write_feather(
        annotations,
        {
            "body": [300, 100, 200],
            "kind": ["output", "input", "interneuron"],
            "side_name": ["right", "left", "left"],
        },
    )
    _write_feather(
        neurotransmitters,
        {
            "body": [100, 200, 300],
            "nt": ["acetylcholine", "gaba", "glutamate"],
        },
    )
    return weights, annotations, neurotransmitters


def _columns() -> MaleCNSColumns:
    return MaleCNSColumns(
        weight_pre="pre_id",
        weight_post="post_id",
        weight_value="strength",
        annotation_id="body",
        annotation_type="kind",
        annotation_side="side_name",
        neurotransmitter_id="body",
        neurotransmitter_name="nt",
    )


def test_local_feather_conversion_is_sorted_and_signed(
    malecns_tables: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    weights, annotations, neurotransmitters = malecns_tables
    destination = tmp_path / "male-cns.scx"
    report = convert_male_cns(
        weights,
        annotations,
        neurotransmitters,
        destination,
        columns=_columns(),
        sign_mapping={"acetylcholine": 1, "gaba": -1, "glutamate": 1},
        block_size=2,
        batch_size=1,
    )

    graph = load_artifact(destination).graph

    assert report.n_neurons == 3
    assert report.n_edges == 3
    assert report.saturated_weights == 0
    assert tuple(neuron.external_id for neuron in graph.neurons) == (100, 200, 300)
    assert graph.blocks[0].rows[0][0].weight == 3
    assert graph.blocks[0].rows[0][1].weight == 2
    assert graph.blocks[0].rows[1][0].weight == -4


def test_missing_neurotransmitter_mapping_is_explicit(
    malecns_tables: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    weights, annotations, neurotransmitters = malecns_tables

    with pytest.raises(DataSchemaError, match="sign mapping"):
        convert_male_cns(
            weights,
            annotations,
            neurotransmitters,
            tmp_path / "missing-sign.scx",
            columns=_columns(),
            sign_mapping={"acetylcholine": 1, "glutamate": 1},
            block_size=2,
        )


def test_excluded_neurotransmitter_edges_are_reported(
    malecns_tables: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    weights, annotations, _ = malecns_tables
    neurotransmitters = tmp_path / "neurotransmitters-with-unclear.feather"
    _write_feather(
        neurotransmitters,
        {
            "body": [100, 200, 300],
            "nt": ["unclear", "gaba", "glutamate"],
        },
    )

    report = convert_male_cns(
        weights,
        annotations,
        neurotransmitters,
        tmp_path / "excluded.scx",
        columns=_columns(),
        sign_mapping={"gaba": -1, "glutamate": 1},
        excluded_neurotransmitters=("unclear",),
        block_size=2,
        batch_size=1,
    )

    artifact = load_artifact(tmp_path / "excluded.scx")
    assert report.n_edges == 1
    assert report.excluded_edges == 2
    assert artifact.graph.edge_count == 1
    assert artifact.manifest.conversion["excluded_neurotransmitters"] == ["unclear"]
    assert artifact.manifest.conversion["excluded_edges"] == 2


def test_full_scope_retains_unannotated_weight_endpoints(
    malecns_tables: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    _, annotations, _ = malecns_tables
    weights = tmp_path / "weights-with-unannotated-endpoint.feather"
    neurotransmitters = tmp_path / "neurotransmitters-with-unannotated-endpoint.feather"
    _write_feather(
        weights,
        {
            "pre_id": [200, 100, 100, 400],
            "post_id": [300, 300, 200, 300],
            "strength": [4, 2, 3, 1],
        },
    )
    _write_feather(
        neurotransmitters,
        {
            "body": [100, 200, 300, 400],
            "nt": ["acetylcholine", "gaba", "glutamate", "acetylcholine"],
        },
    )

    report = convert_male_cns(
        weights,
        annotations,
        neurotransmitters,
        tmp_path / "unannotated-endpoint.scx",
        columns=_columns(),
        sign_mapping={"acetylcholine": 1, "gaba": -1, "glutamate": 1},
        block_size=2,
        batch_size=1,
    )

    artifact = load_artifact(tmp_path / "unannotated-endpoint.scx")
    assert report.n_neurons == 4
    assert report.n_edges == 4
    assert tuple(neuron.external_id for neuron in artifact.graph.neurons) == (100, 200, 300, 400)


def test_annotation_node_filter_excludes_unannotated_endpoints(
    malecns_tables: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    _, annotations, _ = malecns_tables
    weights = tmp_path / "weights-with-unannotated-endpoint.feather"
    neurotransmitters = tmp_path / "neurotransmitters-with-unannotated-endpoint.feather"
    _write_feather(
        weights,
        {
            "pre_id": [200, 100, 100, 400],
            "post_id": [300, 300, 200, 300],
            "strength": [4, 2, 3, 1],
        },
    )
    _write_feather(
        neurotransmitters,
        {
            "body": [100, 200, 300, 400],
            "nt": ["acetylcholine", "gaba", "glutamate", "acetylcholine"],
        },
    )

    report = convert_male_cns(
        weights,
        annotations,
        neurotransmitters,
        tmp_path / "annotated-only.scx",
        columns=_columns(),
        sign_mapping={"acetylcholine": 1, "gaba": -1, "glutamate": 1},
        node_filter="annotations",
        block_size=2,
        batch_size=1,
    )

    artifact = load_artifact(tmp_path / "annotated-only.scx")
    assert report.n_neurons == 3
    assert report.n_edges == 3
    assert tuple(neuron.external_id for neuron in artifact.graph.neurons) == (100, 200, 300)
    assert artifact.manifest.conversion["node_filter"] == "annotations"


def test_quantizer_rejects_or_saturates_int16_overflow() -> None:
    with pytest.raises(QuantizationError):
        WeightQuantizer(numerator=40000).quantize(1)
    assert WeightQuantizer(numerator=40000, overflow="saturate").quantize(1) == 32767


def test_cli_convert_uses_only_explicit_local_paths(
    malecns_tables: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    weights, annotations, neurotransmitters = malecns_tables
    result = CliRunner().invoke(
        app,
        [
            "convert",
            "--weights",
            str(weights),
            "--annotations",
            str(annotations),
            "--neurotransmitters",
            str(neurotransmitters),
            "--output",
            str(tmp_path / "cli.scx"),
            "--annotation-id",
            "body",
            "--annotation-type",
            "kind",
            "--annotation-side",
            "side_name",
            "--neurotransmitter-id",
            "body",
            "--neurotransmitter-name",
            "nt",
            "--weight-pre",
            "pre_id",
            "--weight-post",
            "post_id",
            "--weight-value",
            "strength",
            "--sign-mapping",
            '{"acetylcholine": 1, "gaba": -1, "glutamate": 1}',
            "--block-size",
            "2",
            "--batch-size",
            "1",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "converted=" in result.stdout


def test_inspect_reads_schema_without_rows(malecns_tables: tuple[Path, Path, Path]) -> None:
    weights, _, _ = malecns_tables

    assert feather_columns(weights) == ("pre_id", "post_id", "strength")

    result = CliRunner().invoke(app, ["inspect", "--file", str(weights)])

    assert result.exit_code == 0, result.stdout
    assert "pre_id" in result.stdout
