from langchain_core.language_models.chat_models import BaseChatModel

from .loader import Loader


class OpenRouterLoader(Loader):
    def __call__(
        self,
        model: str,
        temperature: float | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> BaseChatModel | None:

        endpoint = self.has_endpoint(kwargs=kwargs) or "https://openrouter.ai/api/v1"
        api_key = self.get_api_key(api_key, "OPENROUTER_API_KEY")

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=api_key,
            temperature=temperature,
            model=model,
            max_tokens=max_tokens,
            base_url=endpoint,
            **kwargs,
        )
