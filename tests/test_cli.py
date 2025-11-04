from typer.testing import CliRunner

from dbhdistfit import __version__
from dbhdistfit.cli import app

runner = CliRunner()


def test_registry_command_lists_distributions() -> None:
    result = runner.invoke(app, ["registry"])
    assert result.exit_code == 0
    assert "weibull" in result.stdout.lower()
    assert "Complete-form Weibull" in result.stdout


def test_version_option() -> None:
    result = runner.invoke(app, ["--verbose"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
