from langchain_core.language_models.llms import LLM

from .openai import OpenAILoader
from .anthropic import AnthropicLoader
from .llama import LlamaLoader
from .xai import XAILoader

loaders = [
    OpenAILoader(),
    AnthropicLoader(),
    XAILoader(),
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
    
    errors = []

    for loader in loaders:
        try:
            llm = loader(model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens, **kwargs)
        except Exception as e:
            errors.append(e)
            continue

        if llm is not None:
            return llm
        
    
    if not errors:
        error_message = f"LLMLoader could not load a model with the name: '{model}'."
    else:
        accumulated_errors = "\n".join(str(e) for e in errors)
        error_message = (
            f"Failed to load model: '{model}'. Check the model name and your API key.\n" +
            "You can typically set the API key with an enviromnet variable, a function argument or by setting --api-key on the command line\n" + 
            f"See this list of errors for more information:\n{accumulated_errors}"
        )
    
    raise ValueError(error_message)