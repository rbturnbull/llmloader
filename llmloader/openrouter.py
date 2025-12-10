import os
from langchain_core.language_models.llms import LLM

from langchain_openai import ChatOpenAI

from .loader import Loader


class OpenRouterLoader(Loader):
    def __call__(
        self,
        model: str,
        temperature: float | None = None,
        api_key: str = "",
        max_tokens: int = None,
        **kwargs,
    ) -> LLM | None:                                

        api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        api_key = api_key if api_key != "" else None
        open_router_url = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")        

        return ChatOpenAI(
            openai_api_key=api_key,
            base_url=open_router_url,
            temperature=temperature,
            model=model,
            max_tokens=max_tokens,
            **kwargs,
        )
