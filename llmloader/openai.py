from langchain_core.language_models.chat_models import BaseChatModel

from .loader import Loader


class OpenAILoader(Loader):
    def __call__(
        self,
        model: str,
        temperature: float | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> BaseChatModel | None:

        if not model.startswith(('gpt', 'o1-')):
            return None

        from langchain_openai import ChatOpenAI

        if self.has_endpoint(**kwargs):
            return None

        api_key = self.get_api_key(api_key, "OPENAI_API_KEY")

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
