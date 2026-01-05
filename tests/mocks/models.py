import contextlib
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage


@contextlib.contextmanager
def patch_llm_fn(fn_name: str, **kwargs):
    """Context manager for patching LLM functions with mock implementations.

    Creates a mock LLM client that returns an AIMessage with the specified prompt
    content when invoked. This is used to test LLM integrations without making
    actual API calls.

    Args:
        fn_name (str): The fully qualified name of the function/class to patch
            (e.g., 'langchain_openai.ChatOpenAI').
        **kwargs: Additional keyword arguments. Supports:
            prompt (str, optional): The prompt text to return in the mock response.
                Defaults to "Hello, world!".

    Yields:
        tuple: A tuple containing (mock_object, prompt_string) where mock_object
            is the patched mock and prompt_string is the prompt that will be
            returned when the mock is invoked.
    """
    with patch(fn_name) as mock:
        prompt = str(kwargs.get("prompt", "Hello, world!"))
        mock_client = MagicMock()
        mock.return_value = mock_client
        mock_client.invoke.return_value = AIMessage(content=prompt)
        yield mock, prompt


@pytest.fixture()
def prompt():
    """Pytest fixture providing a test prompt string.

    Returns:
        str: A test prompt asking for a haiku about love.
    """
    return "Write me a haiku about love"


@pytest.fixture()
def model_auth():
    """Pytest fixture providing test authentication configurations for all supported models.

    Returns:
        dict: A dictionary mapping model provider names to their configuration
            dictionaries. Each configuration contains the model name and any
            provider-specific parameters needed for testing.
    """
    return {
        "openai": {
            "model": "gpt-5.1",
        },
        "anthropic": {
            "model": "claude-sonnet-4-5",
        },
        "gemini": {
            "model": "gemini-3",
        },
        "xai": {
            "model": "grok-3-pro",
        },
        "mistral": {
            "model": "mistral-3-large",
        },
        "llama": {
            "model": "meta-llama/Llama-3-70b",
        },
        "azure": {
            "model": "deployed_model_name",
            "model_provider": "azure_ai",
        },
        "openrouter": {
            "model": "openrouter/router_model",
        },
    }


@pytest.fixture()
def openai_mock_setup(prompt):
    """Pytest fixture providing mocked OpenAI ChatOpenAI setup.

    Args:
        prompt (str): The test prompt fixture.

    Yields:
        tuple: A tuple containing (mock_object, prompt_string) for testing
            OpenAI model integration.
    """
    with patch_llm_fn("langchain_openai.ChatOpenAI", prompt=prompt) as (mock, prompt):
        yield mock, prompt


@pytest.fixture()
def anthropic_mock_setup(prompt):
    """Pytest fixture providing mocked Anthropic ChatAnthropic setup.

    Args:
        prompt (str): The test prompt fixture.

    Yields:
        tuple: A tuple containing (mock_object, prompt_string) for testing
            Anthropic model integration.
    """
    with patch_llm_fn("langchain_anthropic.ChatAnthropic", prompt=prompt) as (mock, prompt):
        yield mock, prompt


@pytest.fixture()
def gemini_mock_setup(prompt):
    """Pytest fixture providing mocked Google Gemini ChatGoogleGenerativeAI setup.

    Args:
        prompt (str): The test prompt fixture.

    Yields:
        tuple: A tuple containing (mock_object, prompt_string) for testing
            Gemini model integration.
    """
    with patch_llm_fn("langchain_google_genai.ChatGoogleGenerativeAI", prompt=prompt) as (mock, prompt):
        yield mock, prompt


@pytest.fixture()
def xai_mock_setup(prompt):
    """Pytest fixture providing mocked XAI ChatXAI setup.

    Args:
        prompt (str): The test prompt fixture.

    Yields:
        tuple: A tuple containing (mock_object, prompt_string) for testing
            XAI model integration.
    """
    with patch_llm_fn("langchain_xai.ChatXAI", prompt=prompt) as (mock, prompt):
        yield mock, prompt


@pytest.fixture()
def mistral_mock_setup(prompt):
    """Pytest fixture providing mocked Mistral ChatMistralAI setup.

    Args:
        prompt (str): The test prompt fixture.

    Yields:
        tuple: A tuple containing (mock_object, prompt_string) for testing
            Mistral model integration.
    """
    with patch_llm_fn("langchain_mistralai.ChatMistralAI", prompt=prompt) as (mock, prompt):
        yield mock, prompt


@pytest.fixture()
def llama_mock_setup(prompt):
    """Pytest fixture providing mocked Llama ChatLlama3 setup with HuggingFace loader.

    This fixture patches both the HuggingFaceLoader and ChatLlama3 components
    to test Llama model integration without requiring actual model downloads.

    Args:
        prompt (str): The test prompt fixture.

    Yields:
        tuple: A tuple containing (mock_object, prompt_string) for testing
            Llama model integration.
    """
    with patch("llmloader.llama.HuggingFaceLoader.__call__") as hf_mock:

        def init_model(*args, **kwargs):
            model = kwargs.get('model', '')
            return model

        hf_mock.side_effect = init_model
        with patch_llm_fn("llmloader.llama_model.ChatLlama3", prompt=prompt) as (mock, prompt):
            yield mock, prompt


@pytest.fixture()
def azure_mock_setup(prompt):
    """Pytest fixture providing mocked Azure AI AzureAIChatCompletionsModel setup.

    Args:
        prompt (str): The test prompt fixture.

    Yields:
        tuple: A tuple containing (mock_object, prompt_string) for testing
            Azure AI model integration.
    """
    with patch_llm_fn("langchain_azure_ai.chat_models.AzureAIChatCompletionsModel", prompt=prompt) as (mock, prompt):
        yield mock, prompt


@pytest.fixture()
def openrouter_mock_setup(prompt):
    """Pytest fixture providing mocked OpenRouter ChatOpenAI setup.

    OpenRouter uses the OpenAI client with a custom base URL, so this fixture
    patches the ChatOpenAI class to test OpenRouter integration.

    Args:
        prompt (str): The test prompt fixture.

    Yields:
        tuple: A tuple containing (mock_object, prompt_string) for testing
            OpenRouter model integration.
    """
    with patch_llm_fn("langchain_openai.ChatOpenAI", prompt=prompt) as (mock, prompt):
        yield mock, prompt
