from pathlib import Path

from typer.testing import CliRunner

from soup_connectome.cli import app
from soup_connectome.graph.example import example_graph
from soup_connectome.graph.format import write_artifact

runner = CliRunner()


def test_run_example_cli_reports_measured_result() -> None:
    result = runner.invoke(
        app,
        ["run", "--dataset", "example", "--device", "cpu", "--timesteps", "4"],
    )

    assert result.exit_code == 0, result.stdout
    assert "device=cpu" in result.stdout
    assert "measured" in result.stdout


def test_auto_cli_reports_the_selected_backend() -> None:
    result = runner.invoke(
        app,
        ["run", "--dataset", "example", "--device", "auto", "--timesteps", "4"],
    )

    assert result.exit_code == 0, result.stdout
    assert "device=cpu" in result.stdout
    assert "requested_device=auto" in result.stdout


def test_cli_cuda_never_falls_back_to_cpu() -> None:
    result = runner.invoke(app, ["run", "--dataset", "example", "--device", "cuda"])

    if result.exit_code == 0:
        assert "device=cuda" in result.stdout
        assert "requested_device=cuda" in result.stdout
    else:
        assert "cuda" in result.stdout.lower()
        assert "cpu" not in result.stdout.lower()


def test_cli_uses_disk_backed_loader_for_streamed_artifact(tmp_path: Path) -> None:
    artifact_path = write_artifact(example_graph(), tmp_path / "example.scx")

    result = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(artifact_path),
            "--device",
            "cpu",
            "--residency",
            "streamed",
            "--timesteps",
            "4",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "residency=streamed" in result.stdout
