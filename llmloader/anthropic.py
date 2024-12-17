from langchain_core.language_models.llms import LLM
from .loader import Loader

class AnthropicLoader(Loader):
    def __call__(
        model:str,
        temperature:float|None=None, 
        api_key:str="",
        max_tokens:int=None,
        **kwargs
    ) -> LLM|None:
        if not model.startswith('claude'):
            return None
        
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_tokens=max_tokens,
            **kwargs,
        )
