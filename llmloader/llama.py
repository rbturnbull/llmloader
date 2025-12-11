from langchain_core.language_models.llms import LLM
from .huggingface import HuggingFaceLoader


class LlamaLoader(HuggingFaceLoader):
    def __call__(
        self, model: str, temperature: float | None = None, api_key: str = "", max_tokens: int = None, **kwargs
    ) -> LLM | None:
        if model.startswith('meta-llama/Meta-Llama') or model.startswith('meta-llama/Llama'):
            llm = super().__call__(
                model=model,
                temperature=temperature,
                api_key=api_key,
                max_tokens=max_tokens,
                **kwargs,
            )
            from .llama_model import ChatLlama3
            return ChatLlama3(llm=llm)
        return None
