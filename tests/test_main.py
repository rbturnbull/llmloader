
from typer.testing import CliRunner
from llmloader.main import app
from langchain_core.messages import AIMessage

runner = CliRunner()

def test_main():
    result = runner.invoke(app, [
        "Write me a haiku about love",
        "--model",
        "dummy",
    ])
    assert result.exit_code == 0, f"{result.stdout}, {result.exception}"
    assert result.stdout.strip() == "Write me a haiku about love"


def test_azure(azure_mock_setup):    
    prompt = "Write me a haiku about love"
    _, prompt = azure_mock_setup

    result = runner.invoke(app, [
        prompt,
        "--model",
        "gpt-4.1-nano",
    ])    

    assert not result.exception
    assert result.exit_code == 0, f"{result.stdout}, {result.exception}"
    assert result.stdout.strip() == prompt