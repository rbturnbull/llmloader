import llmloader
from langchain_core.messages import AIMessage

def test_azure():
    llm = llmloader.load("gpt-4.1-nano")
    result = llm.invoke("Write me a haiku about love")
    assert isinstance(result, AIMessage)
    assert result.content is not None and len(result.content) > 0    