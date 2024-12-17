from langchain_core.language_models.llms import LLM
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatResult,
    HumanMessage,
    SystemMessage,
)
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from .huggingface import HuggingFaceLoader


class ChatLlama3(BaseChatModel):
    llm:HuggingFacePipeline

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "ChatLlama3"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str]|None = None,
        run_manager: CallbackManagerForLLMRun|None = None,
        **kwargs,
    ) -> ChatResult:
        llama_messages = []

        for message in messages:
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                role = ""
            llama_messages.append(dict(role=role, content=message.content))

        terminators = [
            self.llm.pipeline.tokenizer.eos_token_id,
            self.llm.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.llm.pipeline(
            llama_messages,
            # max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            # temperature=0.1,
            # top_p=0.2,
        )
        result = outputs[0]["generated_text"][-1]
        chat_generations = []

        chat_generation = ChatGeneration(
            message=AIMessage(content=result['content'])#, generation_info=g.generation_info
        )
        chat_generations.append(chat_generation)

        return ChatResult(
            generations=chat_generations, #llm_output=result['content']
        )    


class LlamaLoader(HuggingFaceLoader):
    def __call__(
        self,            
        model:str,
        temperature:float|None=None, 
        api_key:str="",
        max_tokens:int=None,
        **kwargs
    ) -> LLM|None:
        if model.startswith('meta-llama/Meta-Llama') or model.startswith('meta-llama/Llama'):
            llm = super().__call__(
                model=model,
                temperature=temperature,
                api_key=api_key,
                max_tokens=max_tokens,
                **kwargs,
            )
            return ChatLlama3(llm=llm)
        return None
