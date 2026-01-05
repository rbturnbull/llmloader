from langchain_core.language_models.chat_models import BaseChatModel

from .loader import Loader


class GeminiLoader(Loader):
    def __call__(
        self,
        model: str,
        temperature: float | None = None,
        api_key: str | None = "",
        max_tokens: int | None = None,
        **kwargs,
    ) -> BaseChatModel | None:

        if not model.startswith(("gemini", "gemma")):
            return None

        if self.has_endpoint(kwargs=kwargs):
            return None

        api_key = self.get_api_key(api_key, "GOOGLE_API_KEY")
        temperature = temperature or 1.0

        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
