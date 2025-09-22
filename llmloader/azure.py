from langchain_core.language_models.llms import LLM

from .loader import Loader


class AzureOpenAILoader(Loader):
    def __call__(
        self,
        model: str = "",
        temperature: float | None = None,
        api_key: str = "",
        api_version: str = "2024-12-01-preview",
        max_tokens: int = None,
        **kwargs,
    ) -> LLM | None:
        
        from langchain_openai import AzureChatOpenAI
        from dotenv import load_dotenv
        import os

        load_dotenv()

        api_key = api_key if api_key else os.getenv("AZURE_OPENAI_API_KEY")
        api_version = api_version if api_version else os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")     
           
        return AzureChatOpenAI(
            azure_deployment=model,
            api_key=api_key,            
            api_version=api_version,
            azure_endpoint=api_endpoint,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )