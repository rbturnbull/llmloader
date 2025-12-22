from langchain_core.language_models.chat_models import BaseChatModel
from .huggingface import HuggingFaceLoader


class LlamaLoader(HuggingFaceLoader):
    def __call__(
        self,
        model: str,
        temperature: float | None = None,
        api_key: str | None = "",
        max_tokens: int | None = None,
        **kwargs
    ) -> BaseChatModel | None:

        if not model.startswith(('meta-llama/Meta-Llama', 'meta-llama/Llama')):
            return None

        if self.has_endpoint(**kwargs):
            return None

        llm = super().__call__(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_tokens=max_tokens,
            **kwargs,
        )

        from .llama_model import ChatLlama3

        return ChatLlama3(llm=llm)
