from langchain_core.language_models.chat_models import BaseChatModel

from .loader import Loader


class XAILoader(Loader):
    def __call__(
        self,
        model: str,
        temperature: float | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> BaseChatModel | None:

        if not model.startswith('grok'):
            return None

        if self.has_endpoint(**kwargs):
            return None

        api_key = self.get_api_key(api_key, "XAI_API_KEY")

        from langchain_xai import ChatXAI

        return ChatXAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
