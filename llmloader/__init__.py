from langchain_core.language_models.llms import LLM

from .openai import OpenAILoader
from .anthropic import AnthropicLoader
from .llama import LlamaLoader

loaders = [
    OpenAILoader(),
    AnthropicLoader(),
    LlamaLoader(),
]

def load(
    model:str, 
    temperature:float|None=None, 
    api_key:str="",
    max_tokens:int=None,
    **kwargs
) -> LLM:
    # If the model isn't a string, then assume it can work as an LLM
    # This is useful for when the model is already loaded and for testing mock LLMs
    if not isinstance(model, str):
        return model
    
    for loader in loaders:
        try:
            llm = loader(model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens, **kwargs)
        except Exception as e:
            print(f"Error loading model: {e}")
            continue

        if llm is not None:
            return llm
        
    raise ValueError(f"Unsupported model: {model}")