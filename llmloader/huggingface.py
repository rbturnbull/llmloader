import os
from langchain_core.language_models.llms import LLM

from .loader import Loader


class HuggingFaceLoader(Loader):
    def __call__(
        self,            
        model:str,
        temperature:float|None=None, 
        api_key:str="",
        max_tokens:int=None,
        **kwargs
    ) -> LLM|None:
        """ Adapted from https://www.pinecone.io/learn/llama-2/ """

        model_name = model

        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        import torch
        import transformers

        torch.cuda.empty_cache()

        # from langchain_huggingface import HuggingFacePipeline
        from langchain_community.llms import HuggingFacePipeline

        if not api_key:
            api_key = os.getenv('HF_AUTH')

        if not max_tokens:
            max_tokens = 1024

        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='fp4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        # begin initializing HF items, need auth token for these
        model_config = transformers.AutoConfig.from_pretrained(
            model_name,
            token=api_key,
        )

        # initialize the model
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            token=api_key,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )
        model.eval()

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            token=api_key,
        )

        # device = 0 if torch.cuda.is_available() else -1
        pipeline = transformers.pipeline(
            model=model, 
            tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            repetition_penalty=1.1,  # without this output begins repeating
            max_new_tokens=max_tokens,
        )

        llm = HuggingFacePipeline(pipeline=pipeline)

        return llm

