from langchain_core.language_models.chat_models import BaseChatModel

from .loader import Loader


class AzureAILoader(Loader):
    def __call__(
        self,
        model: str,
        temperature: float | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> BaseChatModel | None:

        if "/" in model:
            return None

        endpoint = self.has_endpoint(**kwargs)
        credential = self.get_api_key(api_key)
        kwargs["model_provider"] = "azure_ai"

        from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

        return AzureAIChatCompletionsModel(
            model=model,
            credential=credential,
            endpoint=endpoint,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
