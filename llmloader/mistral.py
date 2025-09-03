from langchain_core.language_models.llms import LLM
from .loader import Loader


class MistralLoader(Loader):
    def __call__(
        self,
        model: str,
        temperature: float | None = None,
        api_key: str = "",
        max_tokens: int = None,
        **kwargs,
    ) -> LLM | None:
        if not model.startswith(('mistral', 'pixtral', 'codestral', 'ministral', 'open-mistral')):
            return None

        if max_tokens:
            kwargs['max_tokens'] = max_tokens
        if api_key:
            kwargs['api_key'] = api_key

        temperature = temperature or 0.0

        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(
            model=model,
            temperature=temperature,
            **kwargs,
        )
