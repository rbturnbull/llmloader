import pytest

pytest_plugins = ["mocks.models"]


@pytest.fixture()
def credentials(model_auth):
    """Pytest fixture providing complete credential configurations for all models.

    Takes the model authentication configurations and enriches them with common
    base credentials (api_key, temperature, max_tokens) that are shared across
    all model providers for testing purposes.

    Args:
        model_auth (dict): The model_auth fixture containing basic model
            configurations for each provider.

    Returns:
        dict: A dictionary mapping model provider names to their complete
            credential dictionaries, including both provider-specific settings
            and common base credentials.
    """
    base_credentials = {
        "api_key": "key123",
        "temperature": 0.7,
        "max_tokens": 100,
    }
    for authdata in model_auth.values():
        authdata.update(base_credentials)
    return model_auth


@pytest.fixture()
def providers(
    openai_mock_setup,
    anthropic_mock_setup,
    gemini_mock_setup,
    xai_mock_setup,
    mistral_mock_setup,
    llama_mock_setup,
    azure_mock_setup,
    openrouter_mock_setup,
):
    """Pytest fixture that aggregates all model provider mock setups for testing.

    This fixture depends on all individual model mock setup fixtures, ensuring
    that they are all initialized before running tests that require multiple
    model providers. Each provider configuration includes its name, mock setup
    data, and required environment variables.

    Args:
        openai_mock_setup: Fixture for mocking OpenAI model setup.
        anthropic_mock_setup: Fixture for mocking Anthropic model setup.
        gemini_mock_setup: Fixture for mocking Gemini model setup.
        xai_mock_setup: Fixture for mocking XAI model setup.
        mistral_mock_setup: Fixture for mocking Mistral model setup.
        llama_mock_setup: Fixture for mocking LLaMA model setup.
        azure_mock_setup: Fixture for mocking Azure model setup.
        openrouter_mock_setup: Fixture for mocking OpenRouter model setup.

    Returns:
        list: A list of tuples, where each tuple contains:
            - str: Provider name (e.g., "openai", "anthropic").
            - tuple: Mock setup data from the corresponding fixture.
            - list: Required environment variable names for the provider.
    """
    data = [
        ("openai", openai_mock_setup, ["OPENAI_API_KEY"]),
        ("anthropic", anthropic_mock_setup, ["ANTHROPIC_API_KEY"]),
        ("gemini", gemini_mock_setup, ["GOOGLE_API_KEY"]),
        ("xai", xai_mock_setup, ["XAI_API_KEY"]),
        ("mistral", mistral_mock_setup, ["MISTRAL_API_KEY"]),
        ("llama", llama_mock_setup, ["HF_AUTH"]),
        ("azure", azure_mock_setup, ["CUSTOM_API_KEY", "CUSTOM_ENDPOINT"]),
        ("openrouter", openrouter_mock_setup, ["OPENROUTER_API_KEY"]),
    ]
    return data


@pytest.fixture()
def azure_llm_mock():
    """Pytest fixture providing a mock Azure AI LLM instance for wrapper testing.

    Returns:
        MagicMock: A mock LLM instance with class name 'AzureAIChatCompletionsModel'.
    """
    from unittest.mock import MagicMock

    mock_llm = MagicMock()
    mock_llm.__class__.__name__ = "AzureAIChatCompletionsModel"
    return mock_llm


@pytest.fixture()
def openai_llm_mock():
    """Pytest fixture providing a mock OpenAI LLM instance for wrapper testing.

    Returns:
        MagicMock: A mock LLM instance with class name 'ChatOpenAI'.
    """
    from unittest.mock import MagicMock

    mock_llm = MagicMock()
    mock_llm.__class__.__name__ = "ChatOpenAI"
    return mock_llm


@pytest.fixture()
def anthropic_llm_mock():
    """Pytest fixture providing a mock Anthropic LLM instance for wrapper testing.

    Returns:
        MagicMock: A mock LLM instance with class name 'ChatAnthropic'.
    """
    from unittest.mock import MagicMock

    mock_llm = MagicMock()
    mock_llm.__class__.__name__ = "ChatAnthropic"
    return mock_llm


@pytest.fixture()
def token_response_metadata():
    """Pytest fixture providing sample response metadata with token usage.

    Returns:
        dict: A dictionary containing token usage information with 50 input tokens,
            100 output tokens, and 150 total tokens.
    """
    return {
        "token_usage": {
            "input_tokens": 50,
            "output_tokens": 100,
            "total_tokens": 150,
        }
    }
