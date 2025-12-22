from langchain_core.language_models.chat_models import BaseChatModel
from .loader import Loader


class MistralLoader(Loader):
    def __call__(
        self,
        model: str,
        temperature: float | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> BaseChatModel | None:
        
        if not model.startswith(('mistral', 'pixtral', 'codestral', 'ministral', 'open-mistral')):
            return None        
        
        if self.has_endpoint(**kwargs):
            return None
        
        api_key = self.get_api_key(api_key, "MISTRAL_API_KEY")

        temperature = temperature or 0.0

        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
