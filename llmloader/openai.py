from langchain_core.language_models.llms import LLM

from .loader import Loader

class OpenAILoader(Loader):
    def __call__(
        self,            
        model:str,
        temperature:float|None=None, 
        api_key:str="",
        max_tokens:int=None,
        **kwargs
    ) -> LLM|None:
        if not model.startswith('gpt') and not model.startswith('o1-'):
            return None
        
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            openai_api_key=api_key, 
            temperature=temperature,
            model_name=model,
            max_tokens=max_tokens,
            **kwargs,
        )
