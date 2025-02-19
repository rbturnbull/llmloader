from langchain_core.language_models.llms import LLM
from .loader import Loader

class XAILoader(Loader):
    def __call__(
        self,
        model:str,
        temperature:float|None=None, 
        api_key:str="",
        max_tokens:int=None,
        **kwargs
    ) -> LLM|None:
        if not model.startswith('grok'):
            return None
        
        if max_tokens:
            kwargs['max_tokens'] = max_tokens
        if api_key:
            kwargs['api_key'] = api_key

        from langchain_xai import ChatXAI
        return ChatXAI(
            model=model,
            temperature=temperature,
            **kwargs,
        )
