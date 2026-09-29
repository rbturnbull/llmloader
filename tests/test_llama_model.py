from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, ChatMessage, HumanMessage, SystemMessage

from llmloader.llama_model import ChatLlama3


def make_chat_llama(reply: str = "Hello from Llama"):
    """Test helper that builds a ChatLlama3 around a mocked HuggingFace pipeline.

    Args:
        reply: The content the mocked pipeline returns as the assistant's message.

    Returns:
        tuple: (chat_model, pipeline_mock)
    """
    pipeline_mock = MagicMock()
    pipeline_mock.tokenizer.eos_token_id = 1
    pipeline_mock.tokenizer.convert_tokens_to_ids.return_value = 2
    pipeline_mock.return_value = [{"generated_text": [{"role": "assistant", "content": reply}]}]
    hf_llm = MagicMock()
    hf_llm.pipeline = pipeline_mock
    return ChatLlama3.model_construct(llm=hf_llm), pipeline_mock


def test_llm_type():
    """Test that ChatLlama3 reports its LLM type."""
    chat, _ = make_chat_llama()
    assert chat._llm_type == "ChatLlama3"


def test_invoke_maps_roles():
    """Test that ChatLlama3 converts LangChain messages to Llama roles and returns an AIMessage."""
    chat, pipeline_mock = make_chat_llama("Hello from Llama")

    result = chat.invoke(
        [
            SystemMessage(content="system prompt"),
            HumanMessage(content="user prompt"),
            AIMessage(content="assistant reply"),
            ChatMessage(role="tool", content="other message"),
        ]
    )

    assert isinstance(result, AIMessage)
    assert result.content == "Hello from Llama"
    llama_messages = pipeline_mock.call_args.args[0]
    assert [m["role"] for m in llama_messages] == ["system", "user", "assistant", ""]
    assert [m["content"] for m in llama_messages] == [
        "system prompt",
        "user prompt",
        "assistant reply",
        "other message",
    ]
    assert pipeline_mock.call_args.kwargs["eos_token_id"] == [1, 2]
    pipeline_mock.tokenizer.convert_tokens_to_ids.assert_called_once_with("<|eot_id|>")
