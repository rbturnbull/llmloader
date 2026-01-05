from pathlib import Path

import yaml
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

__all__ = ["LLMWrapper"]


class LLMWrapper:
    """Wrapper class for handling LLM-specific data formatting and token tracking.

    This class provides utilities for formatting different data types (like images)
    according to the requirements of specific LLM implementations, and for tracking
    token usage across requests.
    """

    IMAGE_FORMATTERS = {
        "AzureAIChatCompletionsModel": lambda data: {
            "type": "image_url",
            "image_url": {"url": f"data:{data.get('mime_type', '')};base64,{data.get('data', '')}"},
        },
        "default": lambda data: {
            "type": "image",
            "source_type": "base64",
            "data": data.get("data", ""),
            "mime_type": data.get("mime_type", ""),
        },
    }

    DATA_TYPES = {
        "image": IMAGE_FORMATTERS,
    }

    @staticmethod
    def format(llm: BaseChatModel, data_type: str, data: dict) -> dict:
        """Format data according to the specific LLM's requirements.

        Args:
            llm: The language model instance to format data for.
            data_type: The type of data to format (e.g., "image").
            data: The data dictionary containing the information to format.

        Returns:
            A formatted dictionary compatible with the specified LLM.

        Raises:
            ValueError: If no formatters are found for the specified data type.
        """
        llm_class_name = llm.__class__.__name__
        formatters = LLMWrapper.DATA_TYPES.get(data_type, {})
        if not formatters:
            raise ValueError(f"No formatters found for data type: {data_type}")
        formatter = formatters.get(llm_class_name, None)
        if formatter is None:
            formatter = LLMWrapper.IMAGE_FORMATTERS["default"]
        return formatter(data)

    @staticmethod
    def get_token_count(response_metadata: AIMessage | dict, record: Path | str = "") -> dict:
        """Track and accumulate token usage to a YAML file.

        Extracts token usage information from response metadata and optionally
        accumulates it to a persistent YAML record file. If a record file path
        is provided, reads existing token counts, adds the new usage, and writes
        the updated totals back to the file.

        Args:
            response_metadata: Dictionary containing token usage information
                with a "token_usage" key that includes "input_tokens",
                "output_tokens", and "total_tokens".
            record: Optional path to the YAML file where token counts are stored.
                If empty string or not provided, no file persistence occurs.
                Defaults to "".

        Returns:
            A dictionary containing the token usage from the current response
            with keys "input_tokens", "output_tokens", and "total_tokens".
        """
        response_metadata = (
            response_metadata.response_metadata if isinstance(response_metadata, AIMessage) else response_metadata
        )
        token_usage = response_metadata.get("token_usage", {})
        new_record = {
            "input_tokens": token_usage.get("input_tokens", 0) or token_usage.get("prompt_tokens", 0),
            "output_tokens": token_usage.get("output_tokens", 0) or token_usage.get("completion_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0) or token_usage.get("all_tokens", 0),
        }
        if record:
            record = Path(record)
            data = dict()
            if record.exists():
                with open(record, "r") as f:
                    data = yaml.safe_load(f) or dict()
            for key, value in new_record.items():
                data[key] = value + data.get(key, 0)
            record.parent.mkdir(parents=True, exist_ok=True)
            with open(record, "w") as f:
                yaml.dump(data, f)
        return new_record
