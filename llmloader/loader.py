from abc import ABC, abstractmethod
from os import getenv
import warnings
from langchain_core.language_models.chat_models import BaseChatModel


class Loader(ABC):
    @abstractmethod
    def __call__(
        self,
        model: str,
        temperature: float | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> BaseChatModel | None:
        raise NotImplementedError

    def has_endpoint(self, key_env: str = "CUSTOM_ENDPOINT", **kwargs) -> str | None:
        endpoint = kwargs.get("endpoint", getenv(key_env, None))
        if endpoint:
            warnings.warn(
                "A custom endpoint is set, ignoring loaders except Azure AI and OpenRouter. "
                "If this was not intended, do not pass an argument to `endpoint` parameter and ensure that CUSTOM_ENDPOINT is not set in your environment variables.",
                UserWarning,
                stacklevel=2,
            )
        return endpoint

    def get_api_key(self, api_key: str | None = "", key_env: str = "CUSTOM_API_KEY") -> str | None:
        return api_key or getenv(key_env, None)
