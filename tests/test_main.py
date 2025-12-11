
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


def test_azure(force_azure_by_fail_openai, monkeypatch):    
    prompt = force_azure_by_fail_openai
    # Set Azure environment variable to trigger AzureOpenAILoader
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://dummy-azure-endpoint.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummykey123")
    
    result = runner.invoke(app, [
        prompt,
        "--model",
        "gpt-4.1-nano",
    ])            
    assert not result.exception, f"Exception occurred: {result.exception}"
    assert result.exit_code == 0, f"{result.stdout}, {result.exception}"
    assert result.stdout.strip() == prompt

def test_gemini_call(gemini_mock_setup, monkeypatch):
    _, prompt = gemini_mock_setup

    monkeypatch.setenv("GOOGLE_API_KEY", "dummykey123")

    result = runner.invoke(app, [
        prompt,
        "--model",
        "gemini-1.5-flash",
    ])            
    assert not result.exception, f"Exception occurred: {result.exception}"
    assert result.exit_code == 0, f"{result.stdout}, {result.exception}"
    assert result.stdout.strip() == prompt

def test_openrouter(openrouter_mock_setup, monkeypatch):
    _, prompt = openrouter_mock_setup

    monkeypatch.setenv("OPENROUTER_API_KEY", "dummykey123")

    result = runner.invoke(app, [
        prompt,
        "--model",
        "openrouter-model-name",
    ])
    assert not result.exception, f"Exception occurred: {result.exception}"
    assert result.exit_code == 0, f"{result.stdout}, {result.exception}"
    assert result.stdout.strip() == prompt