from abc import ABC, abstractmethod
from langchain_core.language_models.llms import LLM


class Loader(ABC):    
    @abstractmethod
    def __call__(
        self,
        model:str,
        temperature:float|None=None, 
        api_key:str="",
        max_tokens:int=None,
        **kwargs
    ) -> LLM|None:
        raise NotImplementedError
