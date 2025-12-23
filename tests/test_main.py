from typer.testing import CliRunner
from llmloader.main import app

runner = CliRunner()


def call(model_name: str, prompt: str = "Write me a haiku about love"):
    """Test helper function to invoke the CLI app with a model and prompt.

    Args:
        model_name: Name of the LLM model to use for the test.
        prompt: The prompt to send to the model. Defaults to "Write me a haiku about love".

    Raises:
        AssertionError: If the command exits with non-zero code or output doesn't match expected prompt.
    """
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
    """Test the CLI app with the dummy model provider."""
    call("dummy")


def test_call_providers(providers, monkeypatch):
    """Test the CLI app with multiple provider configurations.

    Args:
        providers: Pytest fixture containing list of provider configurations.
            Each configuration is a tuple of (name, mock_setup, env_vars).
        monkeypatch: Pytest fixture for safely setting environment variables.
    """
    for name, mock_setup, env_vars in providers:
        _, prompt = mock_setup
        for env_var in env_vars:
            monkeypatch.setenv(env_var, "dummyenv")
        call(name, prompt)