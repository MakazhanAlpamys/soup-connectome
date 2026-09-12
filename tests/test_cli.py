from typer.testing import CliRunner

from soup_connectome.cli import app

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


def test_cli_rejects_unimplemented_cuda() -> None:
    result = runner.invoke(app, ["run", "--dataset", "example", "--device", "cuda"])

    assert result.exit_code != 0
    assert "cuda" in result.stdout.lower()
