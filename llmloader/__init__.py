from langchain_core.language_models.llms import LLM

from .openai import OpenAILoader
from .anthropic import AnthropicLoader
from .llama import LlamaLoader

loaders = [
    OpenAILoader(),
    AnthropicLoader(),
    LlamaLoader(),
]

def load(model_id:str, temperature:float|None=None, api_key:str="") -> LLM:
    for loader in loaders:
        try:
            model = loader(model_id, api_key=api_key, temperature=temperature)
        except Exception as e:
            continue

        if model is not None:
            return model
        
    raise ValueError(f"Unsupported model: {model_id}")