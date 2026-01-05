from pathlib import Path

import yaml
from langchain_core.language_models.chat_models import BaseChatModel


class LLMWrapper:

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
        llm_class_name = llm.__class__.__name__
        formatters = LLMWrapper.DATA_TYPES.get(data_type, {})
        if not formatters:
            raise ValueError(f"No formatters found for data type: {data_type}")
        formatter = formatters.get(llm_class_name, None)
        if formatter is None:
            formatter = LLMWrapper.IMAGE_FORMATTERS["default"]
        return formatter(data)

    @staticmethod
    def get_token_count(record: Path | str, response_metadata: dict, id: str) -> None:
        token_usage = response_metadata.get("token_usage", {})
        new_record = {
            "input_tokens": token_usage.get("input_tokens", 0),
            "output_tokens": token_usage.get("output_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
        }
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
