from langchain_core.language_models.chat_models import BaseChatModel

from .loader import Loader


class AnthropicLoader(Loader):
    def __call__(
        self,
        model: str,
        temperature: float | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        **kwargs
    ) -> BaseChatModel | None:

        if not model.startswith('claude'):
            return None

        if self.has_endpoint(**kwargs):
            return None

        api_key = self.get_api_key(api_key, "ANTHROPIC_API_KEY")

        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
