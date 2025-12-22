"""Manual integration tests for llmloader with various LLM providers.

This module contains manual tests that require actual API keys and make real API calls
to various LLM providers. These tests are marked with the 'manual' marker and should be
run explicitly when needed to verify integration with external services.
"""

import pytest

pytestmark = pytest.mark.manual

import llmloader, logging
from dotenv import dotenv_values
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

def assert_llm_response(model, llm_type: BaseChatModel, prompt="h"):
    """Test that an LLM model loads correctly and returns a valid response.

    Args:
        model: The model identifier to load via llmloader.
        llm_type: The expected LangChain chat model class type.
        prompt: The test prompt to send to the model. Defaults to "h".

    Raises:
        ValueError: If the LLM fails to invoke or returns an invalid response.
        AssertionError: If the response is invalid or the model type doesn't match.
    """
    try:
        llm = llmloader.load(model)
        response = llm.invoke(prompt)
        assert response is not None and len(response.content) > 0, "LLM did not return a valid response"
        assert isinstance(llm, llm_type), f"LLM is not of type {llm_type}"
        logger.info(f"{llm.__class__.__name__} Response: {response}")
    except Exception as e:
        raise ValueError(f"[red]Error invoking LLM[/]: {e}")

def setenv(monkeypatch, key: str):
    """Set an environment variable from .env file for testing.

    Args:
        monkeypatch: pytest's monkeypatch fixture for setting environment variables.
        key: The environment variable key to set from .env file.
    """
    values = dotenv_values(".env")
    monkeypatch.setenv(key, values.get(key, ""))

def test_openai(monkeypatch):
    """Test OpenAI model loading and response generation.

    Args:
        monkeypatch: pytest fixture for setting environment variables.
    """
    from langchain_openai import ChatOpenAI
    setenv(monkeypatch, "OPENAI_API_KEY")
    assert_llm_response("gpt-4.1-nano", ChatOpenAI)

def test_anthropic(monkeypatch):
    """Test Anthropic model loading and response generation.

    Args:
        monkeypatch: pytest fixture for setting environment variables.
    """
    from langchain_anthropic import ChatAnthropic
    setenv(monkeypatch, "ANTHROPIC_API_KEY")
    assert_llm_response("claude-sonnet-4-5", ChatAnthropic)

def test_gemini(monkeypatch):
    """Test Google Gemini model loading and response generation.

    Args:
        monkeypatch: pytest fixture for setting environment variables.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    setenv(monkeypatch, "GOOGLE_API_KEY")
    assert_llm_response("gemini-2.5-flash", ChatGoogleGenerativeAI)

def test_xai(monkeypatch):
    """Test xAI model loading and response generation.

    Args:
        monkeypatch: pytest fixture for setting environment variables.
    """
    from langchain_openai import ChatOpenAI
    setenv(monkeypatch, "XAI_API_KEY")
    assert_llm_response("xai-forefront-1.5", ChatOpenAI)

def test_mistral(monkeypatch):
    """Test Mistral AI model loading and response generation.

    Args:
        monkeypatch: pytest fixture for setting environment variables.
    """
    from langchain_mistralai import ChatMistralAI
    setenv(monkeypatch, "MISTRAL_API_KEY")
    assert_llm_response("mistral-small-2506", ChatMistralAI)

# Uncomment this to test local Llama models if you have the model downloaded and environment set up
# def test_llama(monkeypatch):
#     from llmloader.llama_model import ChatLlama3
#     setenv(monkeypatch, "HF_AUTH")
#     assert_llm_response("meta-llama/Llama-3.1-8B-Instruct", ChatLlama3)

def test_azure(monkeypatch):
    """Test Azure AI model loading and response generation.

    Args:
        monkeypatch: pytest fixture for setting environment variables.
    """
    from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
    setenv(monkeypatch, "CUSTOM_API_KEY")
    setenv(monkeypatch, "CUSTOM_ENDPOINT")
    assert_llm_response("grok-3-mini", AzureAIChatCompletionsModel)

def test_openrouter(monkeypatch):
    """Test OpenRouter model loading and response generation.

    Args:
        monkeypatch: pytest fixture for setting environment variables.
    """
    from langchain_openai import ChatOpenAI
    setenv(monkeypatch, "OPENROUTER_API_KEY")
    assert_llm_response("openai/gpt-4.1-nano", ChatOpenAI)

def test_cli(monkeypatch):
    """Test the CLI application with a custom model endpoint.

    Verifies that the CLI can successfully invoke an LLM model, process the request,
    and return a valid response containing the expected content.

    Args:
        monkeypatch: pytest fixture for setting environment variables.
    """
    from typer.testing import CliRunner
    from llmloader.main import app
    from rich import print

    runner = CliRunner()

    setenv(monkeypatch, "CUSTOM_API_KEY")
    setenv(monkeypatch, "CUSTOM_ENDPOINT")

    result = runner.invoke(
        app,
        [
            "Write me a haiku about love. Start the haiku with haiku:",
            "--model",
            "gpt-5-mini",
            "--temperature",
            "1"
        ]
    )
    assert result.exit_code == 0, f"{result.stdout}, {result.exception}"
    print(f"[green]LLM Response[/]: {result.stdout}")
    assert "haiku" in result.stdout.lower()

