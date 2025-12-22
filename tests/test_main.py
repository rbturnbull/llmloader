from typer.testing import CliRunner
from llmloader.main import app

runner = CliRunner()


def call(model_name: str, prompt: str = "Write me a haiku about love"):
    result = runner.invoke(
        app,
        [
            prompt,
            "--model",
            model_name,
        ],
    )
    assert result.exit_code == 0, f"{result.stdout}, {result.exception}"
    assert result.stdout.strip() == prompt


def test_call_dummy():
    call("dummy")


def test_call_gemini(gemini_mock_setup, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummykey123")
    _, prompt = gemini_mock_setup
    call("gemini", prompt)


def test_call_azure(azure_mock_setup, monkeypatch):
    monkeypatch.setenv("CUSTOM_API_KEY", "dummykey123")
    monkeypatch.setenv("CUSTOM_ENDPOINT", "https://dummy-azure-endpoint.open")
    _, prompt = azure_mock_setup
    call("azure", prompt)


def test_call_openrouter(openrouter_mock_setup, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummykey123")
    _, prompt = openrouter_mock_setup
    call("openrouter", prompt)
