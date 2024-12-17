import os
from langchain_core.language_models.llms import LLM

from .loader import Loader


class HuggingFaceLoader(Loader):
    def __call__(
        model:str,
        temperature:float|None=None, 
        api_key:str="",
        max_tokens:int=None,
        **kwargs
    ) -> LLM|None:

        import torch
        import transformers
        from langchain_community.llms import HuggingFacePipeline

        """ Adapted from https://www.pinecone.io/learn/llama-2/ """

        if not api_key:
            api_key = os.getenv('HF_AUTH')

        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # begin initializing HF items, need auth token for these
        model_config = transformers.AutoConfig.from_pretrained(
            model,
            token=api_key,
        )

        # initialize the model
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            token=api_key,
            **kwargs
        )
        model.eval()

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model,
            token=api_key,
        )

        pipeline = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=max_tokens,  # max number of tokens to generate in the output
            repetition_penalty=1.1  # without this output begins repeating
        )

        llm = HuggingFacePipeline(pipeline=pipeline)

        return llm

