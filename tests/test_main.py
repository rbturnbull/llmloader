
from typer.testing import CliRunner

from llmloader.main import app

runner = CliRunner()

def test_main():
    result = runner.invoke(app, [
        "Write me a haiku about love",
        "--model",
        "dummy",
    ])
    assert result.exit_code == 0, f"{result.stdout}, {result.exception}"
    assert result.stdout.strip() == "Write me a haiku about love"