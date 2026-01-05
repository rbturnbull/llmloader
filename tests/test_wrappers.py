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


def test_get_token_count_with_record_new_file(token_response_metadata, tmp_path):
    """Test token counting with persistence to a new YAML file.

    Args:
        token_response_metadata: Fixture providing sample response metadata.
        tmp_path: Pytest fixture providing a temporary directory.
    """
    from llmloader.wrappers import LLMWrapper

    record_file = tmp_path / "tokens.yaml"
    result = LLMWrapper.get_token_count(token_response_metadata, record=record_file)

    assert result["input_tokens"] == 50
    assert result["output_tokens"] == 100
    assert result["total_tokens"] == 150

    # Verify file was created and contains correct data
    assert record_file.exists()
    with open(record_file, "r") as f:
        data = yaml.safe_load(f)

    assert data["input_tokens"] == 50
    assert data["output_tokens"] == 100
    assert data["total_tokens"] == 150


def test_get_token_count_with_record_existing_file(token_response_metadata, tmp_path):
    """Test token counting with persistence to an existing YAML file.

    Verifies that token counts are accumulated across multiple calls.

    Args:
        token_response_metadata: Fixture providing sample response metadata.
        tmp_path: Pytest fixture providing a temporary directory.
    """
    from llmloader.wrappers import LLMWrapper

    record_file = tmp_path / "tokens.yaml"

    # Create initial file with existing counts
    initial_data = {
        "input_tokens": 25,
        "output_tokens": 50,
        "total_tokens": 75,
    }
    with open(record_file, "w") as f:
        yaml.dump(initial_data, f)

    # Add new counts
    result = LLMWrapper.get_token_count(token_response_metadata, record=record_file)

    assert result["input_tokens"] == 50
    assert result["output_tokens"] == 100
    assert result["total_tokens"] == 150

    # Verify accumulated counts in file
    with open(record_file, "r") as f:
        data = yaml.safe_load(f)

    assert data["input_tokens"] == 75  # 25 + 50
    assert data["output_tokens"] == 150  # 50 + 100
    assert data["total_tokens"] == 225  # 75 + 150


def test_get_token_count_with_record_nested_path(token_response_metadata, tmp_path):
    """Test token counting with persistence to a nested directory path.

    Verifies that parent directories are created automatically.

    Args:
        token_response_metadata: Fixture providing sample response metadata.
        tmp_path: Pytest fixture providing a temporary directory.
    """
    from llmloader.wrappers import LLMWrapper

    record_file = tmp_path / "nested" / "dir" / "tokens.yaml"
    result = LLMWrapper.get_token_count(token_response_metadata, record=record_file)

    assert result["input_tokens"] == 50
    assert result["output_tokens"] == 100
    assert result["total_tokens"] == 150

    # Verify nested directory and file were created
    assert record_file.exists()
    assert record_file.parent.exists()

    with open(record_file, "r") as f:
        data = yaml.safe_load(f)

    assert data["input_tokens"] == 50
    assert data["output_tokens"] == 100
    assert data["total_tokens"] == 150


def test_get_token_count_multiple_accumulations(tmp_path):
    """Test multiple token count accumulations to verify proper summing.

    Args:
        tmp_path: Pytest fixture providing a temporary directory.
    """
    from llmloader.wrappers import LLMWrapper

    record_file = tmp_path / "tokens.yaml"

    # First call
    metadata1 = {
        "token_usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        }
    }
    LLMWrapper.get_token_count(metadata1, record=record_file)

    # Second call
    metadata2 = {
        "token_usage": {
            "input_tokens": 15,
            "output_tokens": 25,
            "total_tokens": 40,
        }
    }
    LLMWrapper.get_token_count(metadata2, record=record_file)

    # Third call
    metadata3 = {
        "token_usage": {
            "input_tokens": 5,
            "output_tokens": 10,
            "total_tokens": 15,
        }
    }
    result = LLMWrapper.get_token_count(metadata3, record=record_file)

    # Last result should reflect only the third call
    assert result["input_tokens"] == 5
    assert result["output_tokens"] == 10
    assert result["total_tokens"] == 15

    # File should have accumulated totals
    with open(record_file, "r") as f:
        data = yaml.safe_load(f)

    assert data["input_tokens"] == 30  # 10 + 15 + 5
    assert data["output_tokens"] == 55  # 20 + 25 + 10
    assert data["total_tokens"] == 85  # 30 + 40 + 15
