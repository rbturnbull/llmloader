import llmloader

def test_dummy():
    llm = llmloader.load("dummy")
    result = llm.invoke("Write me a haiku about love")
    assert result == "Write me a haiku about love"
