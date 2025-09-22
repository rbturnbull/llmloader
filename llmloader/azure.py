from langchain_core.language_models.llms import LLM
from langchain_openai import AzureChatOpenAI

from .loader import Loader


class AzureOpenAILoader(Loader):
    def __call__(
        self,
        model: str,
        temperature: float | None = None,
        api_key: str = "",
        max_tokens: int = None,
        **kwargs,
    ) -> LLM | None:
        
        from dotenv import load_dotenv
        import os

        load_dotenv()

        # Sets the variables required by Azure LLM deployment
        # This is a custom endpoint deployed through Azure AI Foundry, which can include GPT, Grok, DeepSeek, Mistral, etc.
        api_key = api_key if api_key else os.getenv("AZURE_OPENAI_API_KEY", "")
        api_version = kwargs.pop("api_version", os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"))
        azure_endpoint = kwargs.pop("api_endpoint", os.getenv("AZURE_OPENAI_ENDPOINT", ""))    

        return AzureChatOpenAI(
            azure_deployment=model,
            api_key=api_key,          
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )