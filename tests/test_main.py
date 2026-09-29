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


def mock_load(monkeypatch, content: str = "Mock response"):
    """Test helper that patches llmloader.main.load to return a mock LLM.

    Args:
        monkeypatch: Pytest fixture for patching attributes.
        content: The content of the AIMessage returned by the mock LLM.

    Returns:
        MagicMock: The mock LLM returned by the patched load function.
    """
    from unittest.mock import MagicMock

    from langchain_core.messages import AIMessage

    llm = MagicMock()
    llm.invoke.return_value = AIMessage(
        content=content,
        response_metadata={"token_usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}},
    )
    monkeypatch.setattr("llmloader.main.load", MagicMock(return_value=llm))
    return llm


def test_call_count(monkeypatch):
    """Test that --count prints the token usage of the response."""
    mock_load(monkeypatch)
    result = runner.invoke(app, ["prompt", "--count"])
    assert result.exit_code == 0, result.exception
    assert "Mock response" in result.stdout
    assert "'input_tokens': 3" in result.stdout
    assert "'output_tokens': 5" in result.stdout
    assert "'total_tokens': 8" in result.stdout


def test_call_all_results(monkeypatch):
    """Test that --all-results prints the full response object."""
    mock_load(monkeypatch)
    result = runner.invoke(app, ["prompt", "--all-results"])
    assert result.exit_code == 0, result.exception
    assert "AIMessage" in result.stdout


def test_call_passes_options(monkeypatch):
    """Test that the CLI options are passed through to load."""
    mock_load(monkeypatch)
    result = runner.invoke(
        app,
        [
            "prompt",
            "--model",
            "some-model",
            "--temperature",
            "0.5",
            "--max-tokens",
            "10",
            "--api-key",
            "key123",
            "--endpoint",
            "https://custom.endpoint",
        ],
    )
    assert result.exit_code == 0, result.exception
    from llmloader import main

    main.load.assert_called_once_with(
        model="some-model",
        temperature=0.5,
        api_key="key123",
        max_tokens=10,
        endpoint="https://custom.endpoint",
    )


def test_call_default_temperature_is_none(monkeypatch):
    """Test that no temperature is passed to load unless --temperature is given.

    Some models (e.g. OpenAI reasoning models) reject any temperature other than their default.
    """
    mock_load(monkeypatch)
    result = runner.invoke(app, ["prompt"])
    assert result.exit_code == 0, result.exception
    from llmloader import main

    assert main.load.call_args.kwargs["temperature"] is None


def test_openai_payload_omits_default_temperature():
    """Test that an OpenAI model loaded without a temperature doesn't send one in the request payload."""
    import llmloader

    llm = llmloader.load("gpt-6-luna", api_key="key123", endpoint="")
    payload = llm._get_request_payload("prompt")
    assert "temperature" not in payload
