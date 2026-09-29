import sys
from unittest.mock import MagicMock, patch

import pytest

import llmloader
from llmloader.anthropic import AnthropicLoader
from llmloader.gemini import GeminiLoader
from llmloader.huggingface import HuggingFaceLoader
from llmloader.llama import LlamaLoader
from llmloader.loader import Loader
from llmloader.mistral import MistralLoader
from llmloader.openai import OpenAILoader
from llmloader.xai import XAILoader


@pytest.mark.parametrize(
    "loader,model",
    [
        (OpenAILoader(), "gpt-5.1"),
        (AnthropicLoader(), "claude-sonnet-4-5"),
        (GeminiLoader(), "gemini-3"),
        (XAILoader(), "grok-3-pro"),
        (MistralLoader(), "mistral-3-large"),
        (LlamaLoader(), "meta-llama/Llama-3-70b"),
    ],
)
def test_loader_skips_with_endpoint_kwarg(loader, model):
    """Test that provider loaders return None when an endpoint is passed as a keyword argument."""
    with pytest.warns(UserWarning, match="custom endpoint"):
        assert loader(model=model, endpoint="https://custom.endpoint") is None


@pytest.mark.parametrize("loader", [OpenAILoader(), AnthropicLoader(), XAILoader()])
def test_loader_skips_with_endpoint_env(loader, monkeypatch):
    """Test that provider loaders return None when CUSTOM_ENDPOINT is set in the environment."""
    monkeypatch.setenv("CUSTOM_ENDPOINT", "https://custom.endpoint")
    model = {OpenAILoader: "gpt-5.1", AnthropicLoader: "claude-sonnet-4-5", XAILoader: "grok-3-pro"}[type(loader)]
    with pytest.warns(UserWarning, match="custom endpoint"):
        assert loader(model=model) is None


def test_has_endpoint_sets_env_and_pops_kwarg(monkeypatch):
    """Test that has_endpoint stores the endpoint in the environment and removes it from kwargs."""
    kwargs = {"endpoint": "https://custom.endpoint", "other": 1}
    with pytest.warns(UserWarning):
        endpoint = OpenAILoader().has_endpoint(kwargs=kwargs)
    assert endpoint == "https://custom.endpoint"
    assert kwargs == {"other": 1}
    import os

    assert os.environ["CUSTOM_ENDPOINT"] == "https://custom.endpoint"


def test_get_api_key_prefers_argument(monkeypatch):
    """Test that get_api_key prefers the argument over the environment variable."""
    monkeypatch.setenv("CUSTOM_API_KEY", "envkey")
    loader = OpenAILoader()
    assert loader.get_api_key("argkey") == "argkey"
    assert loader.get_api_key("") == "envkey"


def test_abstract_loader_call_raises():
    """Test that the abstract Loader.__call__ raises NotImplementedError when invoked via super()."""

    class IncompleteLoader(Loader):
        def __call__(self, model, **kwargs):
            return super().__call__(model, **kwargs)

    with pytest.raises(NotImplementedError):
        IncompleteLoader()("model")


def test_load_passes_through_non_string():
    """Test that load returns the object unchanged if it is not a string."""
    llm = MagicMock()
    assert llmloader.load(llm) is llm


def test_load_no_loader_matches(monkeypatch):
    """Test that load raises a ValueError when every loader returns None."""
    monkeypatch.setattr(llmloader, "loaders", [lambda **kwargs: None])
    with pytest.raises(ValueError, match="could not load a model with the name: 'unknown'"):
        llmloader.load("unknown")


def test_load_accumulates_errors(monkeypatch):
    """Test that load reports the errors raised by each loader if none succeed."""

    def failing_loader(**kwargs):
        raise RuntimeError("loader exploded")

    monkeypatch.setattr(llmloader, "loaders", [failing_loader, lambda **kwargs: None])
    with pytest.raises(ValueError) as excinfo:
        llmloader.load("unknown")
    assert "Failed to load model: 'unknown'" in str(excinfo.value)
    assert "loader exploded" in str(excinfo.value)


def test_load_continues_after_error(monkeypatch):
    """Test that load moves on to the next loader after one raises an exception."""
    llm = MagicMock()

    def failing_loader(**kwargs):
        raise RuntimeError("loader exploded")

    monkeypatch.setattr(llmloader, "loaders", [failing_loader, lambda **kwargs: llm])
    assert llmloader.load("unknown") is llm


@pytest.fixture()
def hf_mocks():
    """Pytest fixture that replaces torch, transformers and HuggingFacePipeline with mocks.

    Yields:
        tuple: (torch_mock, transformers_mock, pipeline_class_mock)
    """
    torch_mock = MagicMock()
    transformers_mock = MagicMock()
    with (
        patch.dict(sys.modules, {"torch": torch_mock, "transformers": transformers_mock}),
        patch("langchain_community.llms.HuggingFacePipeline") as pipeline_class_mock,
    ):
        yield torch_mock, transformers_mock, pipeline_class_mock


def test_huggingface_loader(hf_mocks, monkeypatch):
    """Test that HuggingFaceLoader builds a pipeline with default max tokens and the HF_AUTH token."""
    torch_mock, transformers_mock, pipeline_class_mock = hf_mocks
    monkeypatch.setenv("HF_AUTH", "hfkey")

    llm = HuggingFaceLoader()(model="some/model", temperature=0.5)

    assert llm is pipeline_class_mock.return_value
    torch_mock.cuda.empty_cache.assert_called_once()
    transformers_mock.AutoConfig.from_pretrained.assert_called_once_with("some/model", token="hfkey")
    transformers_mock.AutoModelForCausalLM.from_pretrained.return_value.eval.assert_called_once()
    transformers_mock.AutoTokenizer.from_pretrained.assert_called_once_with("some/model", token="hfkey")
    pipeline_kwargs = transformers_mock.pipeline.call_args.kwargs
    assert pipeline_kwargs["temperature"] == 0.5
    assert pipeline_kwargs["max_new_tokens"] == 1024
    pipeline_class_mock.assert_called_once_with(pipeline=transformers_mock.pipeline.return_value)


def test_huggingface_loader_explicit_args(hf_mocks):
    """Test that HuggingFaceLoader uses the api_key and max_tokens arguments when given."""
    _, transformers_mock, _ = hf_mocks

    HuggingFaceLoader()(model="some/model", api_key="argkey", max_tokens=10)

    transformers_mock.AutoConfig.from_pretrained.assert_called_once_with("some/model", token="argkey")
    assert transformers_mock.pipeline.call_args.kwargs["max_new_tokens"] == 10


def test_llama_loader_non_llama_model():
    """Test that LlamaLoader returns None for models that are not Llama models."""
    assert LlamaLoader()(model="gpt-5.1") is None


def test_empty_endpoint_not_passed_to_model(openai_mock_setup):
    """Test that an empty endpoint (as the CLI passes by default) is not forwarded to the model class."""
    mock, _ = openai_mock_setup
    llmloader.load("gpt-6-luna", endpoint="")
    assert "endpoint" not in mock.call_args.kwargs


def test_empty_endpoint_not_passed_to_azure(azure_mock_setup, monkeypatch):
    """Test that an empty endpoint kwarg doesn't clash with the endpoint from the environment for Azure."""
    mock, _ = azure_mock_setup
    monkeypatch.setenv("CUSTOM_ENDPOINT", "https://dummy-azure-endpoint.open")
    with pytest.warns(UserWarning):
        llmloader.load("deployed_model_name", endpoint="")
    assert mock.call_args.kwargs["endpoint"] == "https://dummy-azure-endpoint.open"


def test_openrouter_uses_openrouter_api_key(openrouter_mock_setup, monkeypatch):
    """Test that the OpenRouter loader picks up OPENROUTER_API_KEY from the environment."""
    mock, _ = openrouter_mock_setup
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouterkey")
    llmloader.load("openai/gpt-5-mini", endpoint="")
    assert mock.call_args.kwargs["api_key"] == "openrouterkey"
    assert mock.call_args.kwargs["base_url"] == "https://openrouter.ai/api/v1"


def test_openrouter_prefers_custom_api_key(openrouter_mock_setup, monkeypatch):
    """Test that CUSTOM_API_KEY takes precedence over OPENROUTER_API_KEY."""
    mock, _ = openrouter_mock_setup
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouterkey")
    monkeypatch.setenv("CUSTOM_API_KEY", "customkey")
    llmloader.load("openai/gpt-5-mini")
    assert mock.call_args.kwargs["api_key"] == "customkey"


def test_openrouter_key_not_sent_to_custom_endpoint(openrouter_mock_setup, monkeypatch):
    """Test that OPENROUTER_API_KEY is not sent to a custom endpoint that isn't OpenRouter."""
    mock, _ = openrouter_mock_setup
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouterkey")
    with pytest.warns(UserWarning):
        llmloader.load("openai/gpt-5-mini", endpoint="https://custom.endpoint")
    assert mock.call_args.kwargs["api_key"] != "openrouterkey"
    assert mock.call_args.kwargs["base_url"] == "https://custom.endpoint"
