import pytest

pytestmark = pytest.mark.manual

import llmloader, logging
from dotenv import dotenv_values
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

def assert_llm_response(model, llm_type: BaseChatModel, prompt="h"):
    try:
        llm = llmloader.load(model)
        response = llm.invoke(prompt)
        assert response is not None and len(response.content) > 0, "LLM did not return a valid response"    
        assert isinstance(llm, llm_type), f"LLM is not of type {llm_type}"
        logger.info(f"{llm.__class__.__name__} Response: {response}")
    except Exception as e:
        raise ValueError(f"[red]Error invoking LLM[/]: {e}")
    
def setenv(monkeypatch, key: str):
    values = dotenv_values(".env")
    monkeypatch.setenv(key, values.get(key, ""))

def test_openai(monkeypatch):
    from langchain_openai import ChatOpenAI
    setenv(monkeypatch, "OPENAI_API_KEY")
    assert_llm_response("gpt-4.1-nano", ChatOpenAI)

def test_anthropic(monkeypatch):
    from langchain_anthropic import ChatAnthropic
    setenv(monkeypatch, "ANTHROPIC_API_KEY")
    assert_llm_response("claude-sonnet-4-5", ChatAnthropic)

def test_gemini(monkeypatch):
    from langchain_google_genai import ChatGoogleGenerativeAI
    setenv(monkeypatch, "GOOGLE_API_KEY")
    assert_llm_response("gemini-2.5-flash", ChatGoogleGenerativeAI)

def test_xai(monkeypatch):
    from langchain_openai import ChatOpenAI
    setenv(monkeypatch, "XAI_API_KEY")
    assert_llm_response("xai-forefront-1.5", ChatOpenAI)

def test_mistral(monkeypatch):
    from langchain_mistralai import ChatMistralAI
    setenv(monkeypatch, "MISTRAL_API_KEY")
    assert_llm_response("mistral-small-2506", ChatMistralAI)

# Uncomment this to test local Llama models if you have the model downloaded and environment set up
# def test_llama(monkeypatch):
#     from llmloader.llama_model import ChatLlama3
#     setenv(monkeypatch, "HF_AUTH")
#     assert_llm_response("meta-llama/Llama-3.1-8B-Instruct", ChatLlama3)

def test_azure(monkeypatch):
    from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
    setenv(monkeypatch, "CUSTOM_API_KEY")
    setenv(monkeypatch, "CUSTOM_ENDPOINT")
    assert_llm_response("grok-3-mini", AzureAIChatCompletionsModel)

def test_openrouter(monkeypatch):
    from langchain_openai import ChatOpenAI
    setenv(monkeypatch, "OPENROUTER_API_KEY")
    assert_llm_response("openai/gpt-4.1-nano", ChatOpenAI)