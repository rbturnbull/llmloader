from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml


def test_format_image_azure(azure_llm_mock):
    """Test image formatting for Azure AI models.

    Azure models require image URLs in a specific format with embedded MIME types.

    Args:
        azure_llm_mock: Fixture providing a mocked Azure LLM instance.
    """
    from llmloader.wrappers import LLMWrapper

    data = {
        "data": "base64encodeddata",
        "mime_type": "image/png",
    }
    result = LLMWrapper.format(azure_llm_mock, "image", data)

    assert result["type"] == "image_url"
    assert "image_url" in result
    assert result["image_url"]["url"] == "data:image/png;base64,base64encodeddata"


def test_format_image_default(openai_llm_mock):
    """Test image formatting for models using default format.

    Most models use a standard format with separate fields for data and MIME type.

    Args:
        openai_llm_mock: Fixture providing a mocked OpenAI LLM instance.
    """
    from llmloader.wrappers import LLMWrapper

    data = {
        "data": "base64encodeddata",
        "mime_type": "image/jpeg",
    }
    result = LLMWrapper.format(openai_llm_mock, "image", data)

    assert result["type"] == "image"
    assert result["source_type"] == "base64"
    assert result["data"] == "base64encodeddata"
    assert result["mime_type"] == "image/jpeg"


def test_format_image_anthropic(anthropic_llm_mock):
    """Test image formatting for Anthropic models.

    Anthropic models use the default format which includes separate fields
    for data and MIME type.

    Args:
        anthropic_llm_mock: Fixture providing a mocked Anthropic LLM instance.
    """
    from llmloader.wrappers import LLMWrapper

    data = {
        "data": "imagedata123",
        "mime_type": "image/webp",
    }
    result = LLMWrapper.format(anthropic_llm_mock, "image", data)

    assert result["type"] == "image"
    assert result["source_type"] == "base64"
    assert result["data"] == "imagedata123"
    assert result["mime_type"] == "image/webp"


def test_format_invalid_data_type(openai_llm_mock):
    """Test that formatting raises an error for invalid data types.

    Args:
        openai_llm_mock: Fixture providing a mocked OpenAI LLM instance.
    """
    from llmloader.wrappers import LLMWrapper

    data = {"some": "data"}

    with pytest.raises(ValueError, match="No formatters found for data type"):
        LLMWrapper.format(openai_llm_mock, "invalid_type", data)


def test_get_token_count_basic(token_response_metadata):
    """Test basic token counting without file persistence.

    Args:
        token_response_metadata: Fixture providing sample response metadata.
    """
    from llmloader.wrappers import LLMWrapper

    result = LLMWrapper.get_token_count(token_response_metadata)

    assert result["input_tokens"] == 50
    assert result["output_tokens"] == 100
    assert result["total_tokens"] == 150


def test_get_token_count_empty_metadata():
    """Test token counting with empty metadata."""
    from llmloader.wrappers import LLMWrapper

    result = LLMWrapper.get_token_count({})

    assert result["input_tokens"] == 0
    assert result["output_tokens"] == 0
    assert result["total_tokens"] == 0
