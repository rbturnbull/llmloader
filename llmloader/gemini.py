from langchain_core.language_models.llms import LLM

from langchain_google_genai import ChatGoogleGenerativeAI

from .loader import Loader

class GeminiLoader(Loader):
    def __call__(
        self,
        model: str,
        temperature: float | None = None,
        api_key: str = "",
        max_tokens: int = None,
        **kwargs,
    ) -> LLM | None:                        

        api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        api_key = api_key if api_key != "" else None

        if not model.startswith("gemini") or model.startswith("gemma"):
            return None
        
        if not api_key:
            return None

        return ChatGoogleGenerativeAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )